# -*- coding: utf-8 -*
import torch
import torch.nn as nn
import torch.nn.init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# clip_grad_norm_ for 0.4.0, clip_grad_norm for 0.3.1
import numpy as np
import torch.nn.functional as F

from util.constant import device
from util.wordbigfile import WordBigFile
from torch.autograd import Function

"""模型具体结构说明的代码文件
"""


def get_we_parameter(vocab, w2v_file):
    # w2v的映射
    w2v_reader = WordBigFile(w2v_file)
    ndims = w2v_reader.ndims

    we = []
    # we.append([0]*ndims)
    for i in range(len(vocab)):
        try:
            # 传入单词 获取向量表示 每个词dim 500
            vec = w2v_reader.read_one(vocab.idx2word[i])
        except BaseException:
            vec = np.random.uniform(-1, 1, ndims)
        we.append(vec)
    print(
        'getting pre-trained parameter for word embedding initialization',
        np.shape(we))
    return np.array(we)


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


def xavier_init_fc(fc):
    """Xavier initialization for the fully connected layer
    """
    r = np.sqrt(6.) / np.sqrt(fc.in_features +
                              fc.out_features)
    fc.weight.data.uniform_(-r, r)
    fc.bias.data.fill_(0)


# L1惩罚项
class L1Penalty(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clone()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_variables
        # torch.sign() 符号函数
        grad_input = input.clone().sign().mul(0.0001)
        grad_input += grad_output
        return grad_input


# 我们的模型结构定义
class FGMCD(nn.Module):

    def __init__(self, opt):
        super(FGMCD, self).__init__()
        self.concate = opt.concate
        self.opt = opt
        self.Eiters = 0
        # ===============================Brand Aspects=========================
        self.ba_brand_num = opt.brand_num
        # final embedding dim in common space
        self.ba_common_embedding_size = opt.common_embedding_size
        self.ba_num_aspects = 2000
        # brand one-hot embedding
        # 升维与否此处需要调整
        # self.ba_brand_embeddings = nn.Embedding(self.ba_brand_num, self.ba_common_embedding_size)
        self.ba_brand_embeddings = nn.Embedding(self.ba_brand_num, self.ba_num_aspects)
        # 2000*2048
        self.ba_aspects_embeddings = nn.Parameter(torch.randn(self.ba_num_aspects, self.ba_common_embedding_size),
                                                  requires_grad=True)
        self.ba_dropout = nn.Dropout()
        # ===============================Visual Encoder=========================
        self.ve_rnn_output_size = opt.visual_rnn_size * 2
        self.ve_dropout = nn.Dropout(p=opt.dropout)
        self.ve_visual_norm = opt.visual_norm

        # visual bidirectional rnn encoder
        self.ve_rnn = nn.GRU(
            opt.visual_feat_dim,
            opt.visual_rnn_size,
            batch_first=True,
            bidirectional=True)

        self.ve_attn_w_1 = nn.Linear(opt.visual_feat_dim, int(opt.visual_feat_dim // 4), bias=False)
        self.ve_attn_w_2 = nn.Linear(int(opt.visual_feat_dim // 4), 3, bias=False)
        self.ve_attn_tanh = nn.Tanh()
        # self.ve_attn_softmax_1 = nn.Softmax(dim=1)
        self.ve_attn_softmax_0 = nn.Softmax(dim=0)
        nn.init.xavier_uniform_(self.ve_attn_w_1.weight)
        nn.init.xavier_uniform_(self.ve_attn_w_2.weight)

        # visual 1-d convolutional network
        self.ve_convs1 = nn.ModuleList([
            nn.Conv2d(1, opt.visual_kernel_num, (window_size, self.ve_rnn_output_size), padding=(window_size - 1, 0))
            for window_size in opt.visual_kernel_sizes
        ])

        # visual mapping
        self.ve_mfc_fc1 = nn.Linear(opt.visual_mapping_size[0], opt.visual_mapping_size[1])
        self.ve_mfc_dropout = nn.Dropout(opt.dropout)
        self.ve_mfc_bn_1 = nn.BatchNorm1d(opt.visual_mapping_size[1])
        xavier_init_fc(self.ve_mfc_fc1)

        # 添加注意力机制的部分
        self.ve_fc = nn.Linear(opt.visual_feat_dim, opt.visual_feat_dim)
        # ================================Text Encoder==========================
        self.te_text_norm = opt.text_norm
        self.te_word_dim = opt.word_dim
        self.te_we_parameter = opt.we_parameter
        self.te_rnn_output_size = opt.text_rnn_size * 2
        self.te_dropout = nn.Dropout(p=opt.dropout)
        self.te_embed = nn.Embedding(opt.vocab_size, opt.word_dim)
        self.te_rnn = nn.GRU(
            opt.word_dim,
            opt.text_rnn_size,
            batch_first=True,
            bidirectional=True)

        # 1-d convolutional network
        self.te_convs1 = nn.ModuleList([
            nn.Conv2d(1, opt.text_kernel_num, (window_size, self.te_rnn_output_size), padding=(window_size - 1, 0))
            for window_size in opt.text_kernel_sizes
        ])

        # multi fc layers
        self.te_mfc_fc1 = nn.Linear(opt.text_mapping_size[0], opt.text_mapping_size[1])
        self.te_mfc_dropout = nn.Dropout(opt.dropout)
        self.te_mfc_bn_1 = nn.BatchNorm1d(opt.text_mapping_size[1])
        xavier_init_fc(self.te_mfc_fc1)

        if self.te_word_dim == 500 and self.te_we_parameter is not None:
            self.te_embed.weight.data.copy_(torch.from_numpy(self.te_we_parameter))
        else:
            self.te_embed.weight.data.uniform_(-0.1, 0.1)
        # ===============================Fusion Encoder=========================
        # final embedding dim in common space
        self.fe_common_embedding_size = opt.common_embedding_size
        # visual feature dim after visual_net
        self.fe_visual_mapping_size = opt.visual_mapping_size[1]
        # text feature dim after text_net
        self.fe_text_mapping_size = opt.text_mapping_size[1]
        # fusion style
        # 根据使用的是单模态或者多模态
        if opt.single_modal_visual:
            self.fe_fc = nn.Linear(self.fe_visual_mapping_size, self.fe_common_embedding_size)
        elif opt.single_modal_text:
            self.fe_fc = nn.Linear(self.fe_text_mapping_size, self.fe_common_embedding_size)
        else:
            self.fe_fc = nn.Linear(self.fe_text_mapping_size + self.fe_visual_mapping_size,
                                   self.fe_common_embedding_size)
        xavier_init_fc(self.fe_fc)

        # self.brand_optimizer = torch.optim.Adadelta(params2, lr=1)

    def forward(self, brand_ids,
                videos, videos_origin, vis_lengths, vidoes_mask,
                cap_wids, cap_bows, txt_lengths, cap_mask):
        brand_emb = self.embed_brand(brand_ids)
        vid_emb = self.embed_vis(videos, videos_origin, vis_lengths, vidoes_mask)
        cap_emb = self.embed_txt(cap_wids, cap_bows, txt_lengths, cap_mask)

        post_emb = None
        if self.opt.single_modal_visual:
            post_emb = self.fe_fc(vid_emb)
        elif self.opt.single_modal_text:
            post_emb = self.fe_fc(cap_emb)
        else:
            post_emb = torch.cat((vid_emb, cap_emb), 1)
            post_emb = self.fe_fc(post_emb)

        # post represented as single modal
        # post_emb = cap_emb
        return brand_emb, post_emb

    def embed_brand(self, brand_ids, volatile=True):
        # brand embeddings
        brand_emb = self.ba_brand_embeddings(brand_ids)
        # L1 regularization
        brand_emb = L1Penalty.apply(brand_emb)
        # brand 特征升维
        # 逐元素相乘
        # len(brand_list)*2000*2048
        w_aspects = torch.mul(
            brand_emb.view(brand_ids.shape[0], self.ba_num_aspects, 1)
            .expand(brand_ids.shape[0], self.ba_num_aspects, self.ba_aspects_embeddings.shape[1]),
            self.ba_aspects_embeddings.view(1, self.ba_num_aspects, self.ba_aspects_embeddings.shape[1])
            .expand(brand_ids.shape[0], self.ba_num_aspects, self.ba_aspects_embeddings.shape[1]))
        brand_emb = self.ba_dropout(w_aspects)

        brand_emb = brand_emb.permute((1, 0, 2)).mean(0)
        return brand_emb

    def embed_vis(self, videos, videos_origin, lengths, vidoes_mask, volatile=True):
        # deal with videos and texts
        # video_frames data
        # videos:batchsize * 最大帧数 * 2048
        # videos_origin:batchsize * 2048
        # lengths:视频帧数
        # vidoes_mask:mask

        # Level 1. Global Encoding by Mean Pooling According（level 1 平均值）
        org_out = videos_origin

        # Level 2. Temporal-Aware Encoding by biGRU
        # 视觉gru_init_out维度: batchsize（批样本数） * 最大帧(序列长度) * 2048(2048是hidden_size*num_directions,隐层向量大小乘上方向个数)
        gru_init_out, _ = self.ve_rnn(videos)
        # print("visual feature after bi-gru:", gru_init_out.size())
        mean_gru = torch.zeros(
            gru_init_out.size(0),
            self.ve_rnn_output_size).to(device)
        for i, batch in enumerate(gru_init_out):
            mean_gru[i] = torch.mean(batch[:lengths[i]], 0)
        gru_out = mean_gru
        gru_out = self.ve_dropout(gru_out)  # level 2后面用到了drop_out

        # Level 3. Local-Enhanced Encoding by biGRU-CNN
        # (N,C,F1)    #unsqueeze:第三维加上一维 -> batchsize * 最大帧数 * 1024
        vidoes_mask = vidoes_mask.unsqueeze(
            2).expand(-1, -1, gru_init_out.size(2))
        gru_init_out = gru_init_out * vidoes_mask  # 得到卷积后的值
        con_out = gru_init_out.unsqueeze(1)  # 在第二维加上一维
        con_out = [F.relu(conv(con_out)).squeeze(3) for conv in self.ve_convs1]
        con_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in con_out]
        con_out = torch.cat(con_out, 1)
        con_out = self.ve_dropout(con_out)

        # This expects input x to be of size (b x seqlen x d_feat)
        atten_1 = self.ve_attn_w_2(self.ve_attn_tanh(self.ve_attn_w_1(videos)))
        atten_1 = torch.mean(atten_1, dim=-1, keepdim=True)

        weight = torch.zeros_like(atten_1).to(device)
        for i, batch in enumerate(atten_1):
            weight[i, :lengths[i]] = self.ve_attn_softmax_0(batch[:lengths[i]])
        output = (weight * videos).mean(dim=1)

        # concatenation
        vid_emb = None
        if self.concate == 'full':  # level 1+2+3
            vid_emb = torch.cat((gru_out, con_out, org_out, output), 1)  # size (64L, 8192L)
        elif self.concate == 'reduced':  # wmy
            # level 2+3
            vid_emb = torch.cat((gru_out, con_out, output), 1)  # 6144
            # level 1+2
            # vid_emb = torch.cat((gru_out, org_out, out), 1)
            # level 1+3
            # vid_emb = torch.cat((con_out, org_out, out), 1)
            # level 1
            # vid_emb = torch.cat((org_out, out), 1)
            # level 2
            # vid_emb = gru_out
            # level 3
            # vid_emb = con_out

        # mapping to common space
        vid_emb = self.ve_mfc_fc1(vid_emb)
        vid_emb = self.ve_mfc_bn_1(vid_emb)
        vid_emb = self.ve_mfc_dropout(vid_emb)

        if self.ve_visual_norm:
            vid_emb = l2norm(vid_emb)

        return vid_emb

    def embed_txt(self, cap_wids, cap_bows, lengths, cap_mask, volatile=True):
        # Embed word ids to vectors

        # Level 1. Global Encoding by Mean Pooling According
        # one-hot encoding 一条描述就一个向量表示
        org_out = cap_bows

        # Level 2. Temporal-Aware Encoding by biGRU
        # 按序列中出现的索引号取得相应的词向量表示
        cap_wids = self.te_embed(cap_wids)

        packed = pack_padded_sequence(cap_wids, lengths, batch_first=True)

        # 文本双向gru输出维度 batchsize*sequence_len(序列长度)*1024(表示hidden_size*num_directions)
        gru_init_out, _ = self.te_rnn(packed)
        # Reshape *final* out to (batch_size, hidden_size*num_directions)
        padded = pad_packed_sequence(gru_init_out, batch_first=True)

        gru_init_out = padded[0]
        # print(gru_init_out.size())  e.g torch.Size([128,20,1024])
        gru_out = torch.zeros(padded[0].size(0), self.te_rnn_output_size).to(device)
        for i, batch in enumerate(padded[0]):
            gru_out[i] = torch.mean(batch[:lengths[i]], 0)
        gru_out = self.te_dropout(gru_out)

        # Level 3. Local-Enhanced Encoding by biGRU-CNN
        con_out = gru_init_out.unsqueeze(1)
        con_out = [F.relu(conv(con_out)).squeeze(3) for conv in self.te_convs1]
        con_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in con_out]
        con_out = torch.cat(con_out, 1)
        con_out = self.te_dropout(con_out)

        # concatenation
        cap_emb = None
        if self.concate == 'full':  # level 1+2+3
            cap_emb = torch.cat((org_out, gru_out, con_out), 1)  #
        elif self.concate == 'reduced':  # wmy
            # level 2+3
            cap_emb = torch.cat((gru_out, con_out), 1)
            # level 1+2
            # features = torch.cat((org_out, gru_out), 1)
            # level 1+3
            # features = torch.cat((org_out, con_out), 1)
            # level 1
            # features = org_out
            # level 2
            # features = gru_out
            # level 3
            # features = con_out

        # mapping to common space
        cap_emb = self.te_mfc_fc1(cap_emb)
        cap_emb = self.te_mfc_bn_1(cap_emb)
        cap_emb = self.te_mfc_dropout(cap_emb)

        if self.te_text_norm:
            cap_emb = l2norm(cap_emb)

        return cap_emb
