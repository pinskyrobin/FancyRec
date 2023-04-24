# -*- coding: utf-8 -*
import torch
import torch.nn as nn
import torch.nn.init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# clip_grad_norm_ for 0.4.0, clip_grad_norm for 0.3.1
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F

from util.constant import device
from util.wordbigfile import WordBigFile
from transformers import BertModel, BertConfig
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


def xavier_init_fc(fc, bias=True):
    """Xavier initialization for the fully connected layer
    """
    r = np.sqrt(6.) / np.sqrt(fc.in_features +
                              fc.out_features)
    fc.weight.data.uniform_(-r, r)
    if bias:
        fc.bias.data.fill_(0)


# 全连接模块
class MFC(nn.Module):
    """
    Multi Fully Connected Layers
    """

    def __init__(self, fc_layers, dropout):
        super(MFC, self).__init__()
        # fc layers
        # 视频文字全连接层输出 权值矩阵W维度（拼接的视觉特征维度 * 最终公共空间的嵌入维度）
        self.fc1 = nn.Linear(fc_layers[0], fc_layers[1])
        # dropout
        self.dropout = nn.Dropout(p=dropout)
        # batch normalization
        # self.bn_1 = nn.BatchNorm1d(fc_layers[1])
        self.relu = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        xavier_init_fc(self.fc1)

    def forward(self, inputs):
        features = self.fc1(inputs)
        # batch normalization
        # features = self.bn_1(features)
        features = self.relu(features)
        features = self.dropout(features)
        return features


# 多头目注意力机制
class MultiHeadSelfAttention(nn.Module):
    """Self-attention module by Lin, Zhouhan, et al. ICLR 2017"""

    def __init__(self, w1_row, w1_col, w2_col):
        super(MultiHeadSelfAttention, self).__init__()

        self.w_1 = nn.Linear(w1_row, w1_col, bias=False)
        self.w_2 = nn.Linear(w1_col, w2_col, bias=False)
        self.tanh = nn.Tanh()
        # self.softmax_1 = nn.Softmax(dim=1)

        self.softmax_0 = nn.Softmax(dim=0)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.w_1.weight)
        nn.init.xavier_uniform_(self.w_2.weight)

    def forward(self, x, mask=None):
        # This expects input x to be of size (b x seqlen x d_feat)
        atten_1 = self.w_2(self.tanh(self.w_1(x)))
        atten_1 = torch.mean(atten_1, dim=-1, keepdim=True)

        weight = torch.zeros_like(atten_1).to(device)
        for i, batch in enumerate(atten_1):
            weight[i, :mask[i]] = self.softmax_0(batch[:mask[i]])

        output_1 = (weight * x).mean(dim=1)
        return output_1


# 视频数据多层级编码
class VisualEncoder(nn.Module):
    """
    Section 3.1. Video-side Multi-level Encoding
    """

    def __init__(self, opt):
        super(VisualEncoder, self).__init__()

        self.rnn_output_size = opt.visual_rnn_size * 2
        self.dropout = nn.Dropout(p=opt.dropout)
        self.visual_norm = opt.visual_norm
        self.concate = opt.concate
        self.attn = (opt.fusion_style == 'attn')

        # visual bidirectional rnn encoder
        self.rnn = nn.GRU(
            opt.visual_feat_dim,
            opt.visual_rnn_size,
            batch_first=True,
            bidirectional=True)

        self.atten = MultiHeadSelfAttention(opt.visual_feat_dim, int(opt.visual_feat_dim // 4), 3)

        # visual 1-d convolutional network
        self.convs1 = nn.ModuleList([
            nn.Conv2d(1, opt.visual_kernel_num, (window_size, self.rnn_output_size), padding=(window_size - 1, 0))
            for window_size in opt.visual_kernel_sizes
        ])

        if not self.attn:
            # visual mapping
            self.visual_mapping = MFC(opt.visual_mapping_size, opt.dropout)

        # 添加注意力机制的部分
        # self.fc = nn.Linear(opt.visual_feat_dim, opt.visual_feat_dim)

    def forward(self, videos):
        """Extract video_frames feature vectors."""
        # videos:batchsize * 最大帧数 * 2048,mean_original:batchsize * 2048,视频帧数,mask:batchsize*最大帧数
        videos, videos_origin, lengths, vidoes_mask = videos

        # Level 1. Global Encoding by Mean Pooling According（level 1 平均值）
        org_out = videos_origin

        # Level 2. Temporal-Aware Encoding by biGRU
        # 视觉gru_init_out维度: batchsize（批样本数） * 最大帧(序列长度) * 2048(2048是hidden_size*num_directions,隐层向量大小乘上方向个数)
        gru_init_out, _ = self.rnn(videos)
        # print("visual feature after bi-gru:", gru_init_out.size())
        mean_gru = torch.zeros(
            gru_init_out.size(0),
            self.rnn_output_size).to(device)
        for i, batch in enumerate(gru_init_out):
            mean_gru[i] = torch.mean(batch[:lengths[i]], 0)
        gru_out = mean_gru
        gru_out = self.dropout(gru_out)  # level 2后面用到了drop_out

        # Level 3. Local-Enhanced Encoding by biGRU-CNN
        # (N,C,F1)    #unsqueeze:第三维加上一维 -> batchsize * 最大帧数 * 1024
        vidoes_mask = vidoes_mask.unsqueeze(
            2).expand(-1, -1, gru_init_out.size(2))
        gru_init_out = gru_init_out * vidoes_mask  # 得到卷积后的值
        con_out = gru_init_out.unsqueeze(1)  # 在第二维加上一维
        con_out = [F.relu(conv(con_out)).squeeze(3) for conv in self.convs1]
        con_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in con_out]
        con_out = torch.cat(con_out, 1)
        con_out = self.dropout(con_out)

        # 注意力机制层
        output = self.atten(videos, lengths)

        # concatenation
        features = None
        if self.concate == 'full':  # level 1+2+3
            # features = torch.cat((gru_out, con_out, org_out, output), 1)  # size (64L, 8192L)
            features = torch.cat((gru_out, con_out, output), 1)  # size (64L, 8192L)
        elif self.concate == 'reduced':  # wmy
            # level 2+3
            features = torch.cat((gru_out, con_out, output), 1)  # 6144
            # level 1+2
            # features = torch.cat((gru_out, org_out, out), 1)
            # level 1+3
            # features = torch.cat((con_out, org_out, out), 1)
            # level 1
            # features = torch.cat((org_out, out), 1)
            # level 2
            # features = gru_out
            # level 3
            # features = con_out
        # print("before FC visual feature dim", features.size())

        # mapping to common space
        if not self.attn:
            features = self.visual_mapping(features)
            if self.visual_norm:
                features = l2norm(features)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(VisualEncoder, self).load_state_dict(new_state)


# 文本数据（利用transformer进行处理）多层级编码
class TextTransformersEncoder(nn.Module):
    """
    multi-level encoding
    process text feature with transformers
    """

    def __init__(self, opt):
        super(TextTransformersEncoder, self).__init__()
        self.text_norm = opt.text_norm
        self.hidden_size = opt.text_transformers_hidden_size
        self.concate = opt.concate
        self.attn = (opt.fusion_style == 'attn')
        # initial bert model
        # modify the default configuration
        self.configuration = BertConfig(num_hidden_layers=3, num_attention_heads=12)
        self.pretrained_weights = 'bert-base-uncased'
        self.model = BertModel.from_pretrained(self.pretrained_weights, config=self.configuration)

        # 1-d convolutional network
        self.convs1 = nn.ModuleList([
            nn.Conv2d(1, opt.text_kernel_num, (window_size, self.hidden_size), padding=(window_size - 1, 0))
            for window_size in opt.text_kernel_sizes
        ])

        if not self.attn:
            # multi fc layers
            self.text_mapping = MFC(opt.text_mapping_size, opt.dropout)
        # dropout
        self.dropout = nn.Dropout(p=opt.dropout)

    def forward(self, text):
        cap_bows, tokens, type_ids, mask = text
        # tokens = tokens.to(self.device)
        # type_ids = type_ids.to(self.device)
        # mask = mask.to(self.device)
        # last_hidden_state,  pooler_output(cls token), hidden_states(type: tuple, one for each layer)
        # if need hidden-state outputs, set Param output_hidden_states=True
        if torch.cuda.is_available():
            torch.cuda.memory_summary(device=None, abbreviated=False)
        outputs = self.model(input_ids=tokens, token_type_ids=type_ids,
                             attention_mask=mask)
        # one-hot encoding
        org_out = cap_bows
        # print("org_out:", org_out.size(1))
        # bert word embedding
        # bert_embed = torch.zeros(len(tokens), self.hidden_size).cuda()
        # for i, batch in enumerate(outputs[0]):
        #     bert_embed[i] = torch.mean(batch[:int(torch.sum(mask[i]))], 0)
        # average of last two Hidden-state
        # hidden_sum = torch.stack(outputs[2], 0)[-2:].sum(0)
        # average_last_2_layers = torch.zeros(len(tokens), self.hidden_size).cuda()

        # for i, batch in enumerate(hidden_sum):
        #     average_last_2_layers[i] = torch.mean(batch[:int(torch.sum(mask[i]))], 0)

        # OUTPUT of TRANSFORMERS (without intermediate states)
        last_hidden = outputs[0]  # Tensor (batch_size: 8, sequence_length: 154, hidden_size: 768)
        tf_out = torch.zeros(len(tokens), self.hidden_size).to(device)  # Tensor (batch_size: 8, hidden_size: 768)
        for i, batch in enumerate(last_hidden):
            tf_out[i] = torch.mean(batch[:int(torch.sum(mask[i]))], 0)

        # features by CNN
        con_out = last_hidden.unsqueeze(1)  # Tensor (batch_size: 8, 1, sequence_length: 154, hidden_size: 768)
        con_out = [F.relu(conv(con_out)).squeeze(3) for conv in
                   self.convs1]  # List (3 * Tensor (batch_size: 8, ch: 512, sequence_length: 154: 768))
        con_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in con_out]  # List (3 * Tensor (batch_size: 8, ch: 512))
        con_out = torch.cat(con_out, 1)  # Tensor (batch_size: 8, ch: 512 * 3)
        con_out = self.dropout(con_out)  # Tensor (batch_size: 8, ch: 512 * 3)
        # concatenation
        features = None
        if self.concate == 'full':  # level 1+2+3  4318+768+1536
            # features = torch.cat((org_out, tf_out, con_out), 1)
            features = torch.cat((tf_out, con_out), 1)
        elif self.concate == 'reduced':
            # level 2+3
            features = torch.cat((tf_out, con_out), 1)
            # level 1+3
            # features = torch.cat((org_out, con_out), 1)

        # mapping to common space
        if not self.attn:
            features = self.text_mapping(features)
            if self.text_norm:
                features = l2norm(features)

        return features


# 文本数据（利用bi-gru处理）多层级编码
class TextEncoder(nn.Module):
    """
    Section 3.2. Text-side Multi-level Encoding
    process text feature with bi-gru
    """

    def __init__(self, opt):
        super(TextEncoder, self).__init__()
        self.text_norm = opt.text_norm
        self.word_dim = opt.word_dim
        self.we_parameter = opt.we_parameter
        self.rnn_output_size = opt.text_rnn_size * 2
        self.dropout = nn.Dropout(p=opt.dropout)
        self.concate = opt.concate
        self.attn = (opt.fusion_style == 'attn')

        self.embed = nn.Embedding(opt.vocab_size, opt.word_dim)
        self.rnn = nn.GRU(
            opt.word_dim,
            opt.text_rnn_size,
            batch_first=True,
            bidirectional=True)

        # 1-d convolutional network
        self.convs1 = nn.ModuleList([
            nn.Conv2d(1, opt.text_kernel_num, (window_size, self.rnn_output_size), padding=(window_size - 1, 0))
            for window_size in opt.text_kernel_sizes
        ])

        if not self.attn:
            # multi fc layers
            self.text_mapping = MFC(opt.text_mapping_size, opt.dropout)

        self.init_weights()

    def init_weights(self):
        # 初始化成从flickr语料库预训练的词向量
        if self.word_dim == 500 and self.we_parameter is not None:
            self.embed.weight.data.copy_(torch.from_numpy(self.we_parameter))
        else:
            self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, text, *args):
        # Embed word ids to vectors
        # cap_wids, cap_w2vs, cap_bows, cap_mask = x
        cap_wids, cap_bows, lengths, cap_mask = text

        # Level 1. Global Encoding by Mean Pooling According
        # one-hot encoding 一条描述就一个向量表示
        org_out = cap_bows

        # Level 2. Temporal-Aware Encoding by biGRU
        # 按序列中出现的索引号取得相应的词向量表示
        cap_wids = self.embed(cap_wids)

        packed = pack_padded_sequence(cap_wids, lengths, batch_first=True)

        # 文本双向gru输出维度 batchsize*sequence_len(序列长度)*1024(表示hidden_size*num_directions)
        gru_init_out, _ = self.rnn(packed)
        # Reshape *final* out to (batch_size, hidden_size*num_directions)
        padded = pad_packed_sequence(gru_init_out, batch_first=True)

        gru_init_out = padded[0]
        # print(gru_init_out.size())  e.g torch.Size([128,20,1024])
        gru_out = torch.zeros(padded[0].size(0), self.rnn_output_size).to(device)
        for i, batch in enumerate(padded[0]):
            gru_out[i] = torch.mean(batch[:lengths[i]], 0)
        gru_out = self.dropout(gru_out)

        # Level 3. Local-Enhanced Encoding by biGRU-CNN
        con_out = gru_init_out.unsqueeze(1)
        con_out = [F.relu(conv(con_out)).squeeze(3) for conv in self.convs1]
        con_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in con_out]
        con_out = torch.cat(con_out, 1)
        con_out = self.dropout(con_out)

        # concatenation
        features = None
        if self.concate == 'full':  # level 1+2+3
            # features = torch.cat((org_out, gru_out, con_out), 1)
            features = torch.cat((gru_out, con_out), 1)
        elif self.concate == 'reduced':  # wmy
            # level 2+3
            features = torch.cat((gru_out, con_out), 1)
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
        if not self.attn:
            features = self.text_mapping(features)
            if self.text_norm:
                features = l2norm(features)

        return features


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


# 对Brand进行特征表示的模块
class BrandAspects(nn.Module):
    """used for get Brand embedding in validation/test set"""

    def __init__(self, opt):
        super(BrandAspects, self).__init__()

        self.brand_num = opt.brand_num
        # final embedding dim in common space
        self.common_embedding_size = opt.common_embedding_size
        self.num_aspects = opt.brand_aspect
        # brand one-hot embedding
        # 升维与否此处需要调整
        # self.brand_embeddings = nn.Embedding(self.brand_num, self.common_embedding_size)
        self.brand_embeddings = nn.Embedding(self.brand_num + 1, self.num_aspects)
        # 2000*2048
        self.aspects_embeddings = nn.Parameter(torch.randn(self.num_aspects, self.common_embedding_size),
                                               requires_grad=True)
        self.dropout = nn.Dropout()

    def forward(self, brand_list):
        # len(brand_list)*2000
        brand_weights = self.brand_embeddings(brand_list)
        # L1 regularization
        brand_weights = L1Penalty.apply(brand_weights)
        # brand 特征升维
        # 逐元素相乘
        # len(brand_list)*2000*2048
        w_aspects = torch.mul(
            brand_weights.view(brand_list.shape[0], self.num_aspects, 1)
            .expand(brand_list.shape[0], self.num_aspects, self.aspects_embeddings.shape[1]),
            self.aspects_embeddings.view(1, self.num_aspects, self.aspects_embeddings.shape[1])
            .expand(brand_list.shape[0], self.num_aspects, self.aspects_embeddings.shape[1]))
        w_aspects = self.dropout(w_aspects)
        return w_aspects

        # 不进行升维
        # return brand_weights


# MFB用于对视觉、语言特征进行融合
class MFB(nn.Module):
    """
    Multi-modal Factorized Bi-linear Pooling Fusion.
    """

    def __init__(self, opt):
        super(MFB, self).__init__()
        self.opt = opt
        self.config = {"MFB_K": 5, "MFB_O": self.opt.common_embedding_size, "DROP_OUT": 0.1}
        self.proj_i = nn.Linear(self.opt.visual_mapping_size[1], self.config["MFB_K"] * self.config["MFB_O"])
        self.proj_q = nn.Linear(self.opt.text_mapping_size[1], self.config["MFB_K"] * self.config["MFB_O"])
        self.dropout = nn.Dropout(self.config["DROP_OUT"])
        self.pool = nn.AvgPool1d(kernel_size=self.config["MFB_K"], stride=self.config["MFB_K"])
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.proj_i.weight)
        nn.init.xavier_uniform_(self.proj_q.weight)

    def forward(self, img_feat, txt_feat, exp_in=1):
        """
            img_feat.size() -> (N, C, img_feat_size)
            ques_feat.size() -> (N, C, ques_feat_size)
            z.size() -> (N, MFB_O)
            exp_out.size() -> (N, C, K*O)
        """
        batch_size = img_feat.shape[0]
        img_feat = self.proj_i(img_feat).view(batch_size, 1, -1)  # (N, C, K*O)
        txt_feat = self.proj_q(txt_feat).view(batch_size, 1, -1)  # (N, C, K*O)

        # Hadamard product
        exp_out = img_feat * txt_feat  # (N, C, K*O)
        exp_out = self.dropout(exp_out)  # (N, C, K*O)
        z1 = self.pool(exp_out) * self.config['MFB_K']  # (N, C, O)
        z1 = z1.squeeze()
        # z = torch.sqrt(F.relu(z1)) - torch.sqrt(F.relu(-z1))
        # z = F.normalize(z.view(batch_size, -1))               # (N, C*O)
        # z = z.view(batch_size, -1, self.config['MFB_O'])      # (N, C, O)
        return z1


# 视觉模态和语言模态的特征融合方式
# 可以是全连接融合，只使用视觉特征，只使用语言特征
class FusionEncoder(nn.Module):
    def __init__(self, opt):
        super(FusionEncoder, self).__init__()
        self.opt = opt
        # final embedding dim in common space
        self.common_embedding_size = opt.common_embedding_size
        # visual feature dim after visual_net
        self.visual_mapping_size = opt.visual_mapping_size[1]
        # text feature dim after text_net
        self.text_mapping_size = opt.text_mapping_size[1]
        # fusion style
        # 根据使用的是单模态或者多模态
        if opt.single_modal_visual:
            self.fc = nn.Linear(self.visual_mapping_size, self.common_embedding_size)
        elif opt.single_modal_text:
            self.fc = nn.Linear(self.text_mapping_size, self.common_embedding_size)
        else:
            self.fc = nn.Linear(self.text_mapping_size + self.visual_mapping_size, self.common_embedding_size)
        self.init_weights()

    def forward(self, visual_embs, text_embs):
        # control the style of fusion
        # single-modal only
        if self.opt.single_modal_visual:
            fusion_vt = self.fc(visual_embs)
        elif self.opt.single_modal_text:
            fusion_vt = self.fc(text_embs)
        else:
            fusion_vt = torch.cat((visual_embs, text_embs), 1)
            fusion_vt = self.fc(fusion_vt)
        return fusion_vt

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        xavier_init_fc(self.fc)


# Inspired by SimCLR model
class PrjHeadFusionEncoder(nn.Module):
    def __init__(self, opt):
        super(PrjHeadFusionEncoder, self).__init__()
        self.opt = opt

        # final embedding dim in common space
        self.common_embedding_size = opt.common_embedding_size
        # visual feature dim after visual_net
        self.visual_mapping_size = opt.visual_mapping_size[1]
        # text feature dim after text_net
        self.text_mapping_size = opt.text_mapping_size[1]

        self.fc1 = nn.Linear(self.text_mapping_size + self.visual_mapping_size, 512, bias=False)
        self.fc2 = nn.Linear(512, self.common_embedding_size, bias=True)

        self.projection_head = nn.Sequential(
            self.fc1,
            nn.BatchNorm1d(512),
            nn.ReLU(),
            self.fc2
        )
        self.init_weights()

    def forward(self, visual_embs, text_embs):
        fusion_vt = torch.cat((visual_embs, text_embs), 1)
        if self.opt.prj_head_output:
            return fusion_vt
        else:
            return self.projection_head(fusion_vt)

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        xavier_init_fc(self.fc1, bias=False)
        xavier_init_fc(self.fc2)


# Inspired by Paper "Learning Alignment for Multimodal Emotion Recognition from Speech"
class AttnReductionFusionEncoder(nn.Module):

    def __init__(self, opt):
        super(AttnReductionFusionEncoder, self).__init__()
        self.opt = opt

        # final embedding dim in common space
        self.common_embedding_size = opt.common_embedding_size
        # visual feature dim after visual_net
        self.visual_mapping_size = opt.visual_mapping_size[0]
        # text feature dim after text_net
        self.text_mapping_size = opt.text_mapping_size[0]

        # alpha = softmax(tanh(W1*visual_embs + W2*text_embs + b))
        # alpha size: (batch_size, text_mapping_size, visual_mapping_size)
        # visual_embs size: (batch_size, visual_mapping_size)
        # text_embs size: (batch_size, text_mapping_size)
        self.vis_linear = nn.Linear(1, self.text_mapping_size, bias=False)
        self.text_linear = nn.Linear(1, self.visual_mapping_size, bias=False)
        self.b = nn.Parameter(torch.zeros(self.visual_mapping_size))
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.fusion_visual_linear = nn.Linear(self.text_mapping_size, self.common_embedding_size)
        self.fusion_text_linear = nn.Linear(self.visual_mapping_size, self.common_embedding_size)

        # fusion_vt = W3 * (alpha * visual_embs) + W4 * ((1-alpha) * text_embs)^T
        # self.fusion_linear = nn.Linear(self.text_mapping_size, self.common_embedding_size)

    def forward(self, visual_embs, text_embs):
        _visual_embs = visual_embs.unsqueeze(2)  # (batch_size * v_size * 1)
        _text_embs = text_embs.unsqueeze(2)  # (batch_size * t_size * 1)

        visual_attn = self.vis_linear(_visual_embs).transpose(1, 2)  # (batch_size * t_size * v_size)
        text_attn = self.text_linear(_text_embs)  # (batch_size * t_size * v_size)

        alpha = self.softmax(self.tanh(visual_attn + text_attn + self.b))  # (batch_size * t_size * v_size)

        _visual_embs = visual_embs.unsqueeze(1).repeat(1, self.text_mapping_size, 1)  # (batch_size * t_size * v_size)
        _text_embs = text_embs.unsqueeze(1).repeat(1, self.visual_mapping_size, 1)  # (batch_size * v_size * t_size)

        visual_score = torch.sum(alpha * _visual_embs, dim=2)  # (batch_size * t_size)
        text_score = torch.sum(alpha.transpose(1, 2) * _text_embs, dim=2)  # (batch_size * v_size)

        score = torch.sum(alpha * _visual_embs, dim=2)  # (batch_size * t_size)
        fusion_vt = self.relu(self.fusion_visual_linear(visual_score) + self.fusion_text_linear(text_score))

        # fusion_vt = self.fusion_linear(score)  # (batch_size * common_embedding_size)

        return fusion_vt


# 我们的模型结构定义
class FGMCD(nn.Module):

    def __init__(self, opt):
        # Build Models
        super(FGMCD, self).__init__()
        self.opt = opt
        # brand net
        self.brand_encoding = BrandAspects(opt)
        params1 = list(self.brand_encoding.parameters())
        # visual net
        if not opt.single_modal_text:
            self.vid_encoding = VisualEncoder(opt)
            params1 += list(self.vid_encoding.parameters())
        # text net
        if not opt.single_modal_visual:
            self.text_encoding = None
            self.text_net = opt.text_net
            if self.text_net == 'bi-gru':
                self.text_encoding = TextEncoder(opt)
            elif self.text_net == 'transformers':
                self.text_encoding = TextTransformersEncoder(opt)
            params1 += list(self.text_encoding.parameters())
        # fusion net
        self.fusion_style = opt.fusion_style
        if self.fusion_style == 'fc':
            self.fusion_encoding = FusionEncoder(opt)
        elif self.fusion_style == 'mfb':
            self.fusion_encoding = MFB(opt)
        elif self.fusion_style == 'ph':
            self.fusion_encoding = PrjHeadFusionEncoder(opt)
        elif self.fusion_style == 'attn':
            self.fusion_encoding = AttnReductionFusionEncoder(opt)
        params1 += list(self.fusion_encoding.parameters())
        self.params1 = params1

        self.Eiters = 0

    def forward(self, brand_ids, videos, captions):
        # extract and fuse features
        brand_embs = self.embed_brand(brand_ids)

        if self.opt.single_modal_visual:
            post_embs = self.embed_vis(videos)
        elif self.opt.single_modal_text:
            post_embs = self.embed_txt(captions)
        else:
            vid_emb = self.embed_vis(videos)
            cap_emb = self.embed_txt(captions)
            # cap_emb = nn.parallel.data_parallel(self.text_encoding, inputs=text_data, device_ids=[0, 1, 2, 3, 4])
            # vid_emb = nn.parallel.data_parallel(self.vid_encoding, inputs=videos_data, device_ids=[0, 1, 2, 3, 4])

            # post_embs = torch.cat((vid_emb, cap_emb), 1)
            post_embs = self.fusion_encoding(vid_emb, cap_emb)

        # post represented as single modal
        # post_embs = cap_emb
        return brand_embs, post_embs

    def embed_brand(self, brand_ids, volatile=True):
        brand_ids = brand_ids.to(device)
        # brand embeddings
        brand_embs = self.brand_encoding(brand_ids)
        # w_aspects = nn.parallel.data_parallel(self.brand_encoding, inputs=brand_ids, device_ids=[0, 1, 2, 3, 4])
        brand_embs = brand_embs.permute((1, 0, 2)).mean(0)
        return brand_embs

    def embed_vis(self, vis_data, volatile=True):
        # deal with videos and texts
        # video_frames data
        # frames:batchsize * 最大帧数 * 2048
        # mean_original:batchsize * 2048
        # video_lengths:视频帧数
        # vidoes_mask:mask
        frames, mean_origin, video_lengths, vidoes_mask = vis_data

        frames = frames.to(device)
        mean_origin = mean_origin.to(device)
        vidoes_mask = vidoes_mask.to(device)

        data = (frames, mean_origin, video_lengths, vidoes_mask)

        return self.vid_encoding(data)

    def embed_txt(self, text_data, volatile=True):
        data = None
        if self.text_net == 'bi-gru':
            # caption:batchsize*最长句子,cap_bows:batchsize*7807，句子长度，mask
            captions, cap_bows, lengths, cap_masks = text_data
            if captions is not None:
                # captions = Variable(captions, volatile=volatile)
                captions = captions.to(device)

            if cap_bows is not None:
                # cap_bows = Variable(cap_bows)
                cap_bows = cap_bows.to(device)

            if cap_masks is not None:
                # cap_masks = Variable(cap_masks)
                cap_masks = cap_masks.to(device)
            data = (captions, cap_bows, lengths, cap_masks)

        elif self.text_net == 'transformers':
            cap_bows, tokens, type_ids, masks = text_data
            if cap_bows is not None:
                cap_bows = cap_bows.to(device)
            if tokens is not None:
                tokens = tokens.to(device)
            if type_ids is not None:
                type_ids = type_ids.to(device)
            if masks is not None:
                masks = masks.to(device)
            data = (cap_bows, tokens, type_ids, masks)

        return self.text_encoding(data)

    def state_dict(self):
        state_dict = [
            self.vid_encoding.state_dict(),
            self.text_encoding.state_dict(),
            self.brand_encoding.state_dict(),
            self.fusion_encoding.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.vid_encoding.load_state_dict(state_dict[0])
        self.text_encoding.load_state_dict(state_dict[1])
        self.brand_encoding.load_state_dict(state_dict[2])
        self.fusion_encoding.load_state_dict(state_dict[3])
