# -*- coding: utf-8 -*
import torch
from torch.autograd import Variable
import torch.nn as nn
from basic.constant import device

"""计算loss的代码
"""
# l2正则化
def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X

# 余弦相似度
def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    # l2规范化
    im = l2norm(im)
    s = l2norm(s)
    return im.mm(s.t())


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score

# 欧式距离相似度
def euclidean_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.pow(2).sum(2).t()
    return score    


class LabLoss(nn.Module):
    def __init__(self):
        super(LabLoss, self).__init__()

    def forward(self, brand_embs):
        s = cosine_sim(brand_embs, brand_embs)
        # print(s)
        mask = torch.eye(s.size(0)) > .5
        I = mask
        if torch.cuda.is_available():
            I = I.cuda()
        s.masked_fill_(I, 0)
        # print(s)
        loss_lab = (torch.sum(torch.exp(s))-s.size(0))
        if torch.cuda.is_available():
            loss_lab = loss_lab.cuda()
        return loss_lab/s.size(0)

# 跨模态检索用的tripletloss
class TripletLoss(nn.Module):
    """
    triplet ranking loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False, cost_style='sum', direction='all', loss_fun='mrl'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.cost_style = cost_style
        self.direction = direction
        self.loss_fun = loss_fun
        if measure == 'order':
            self.sim = order_sim
        elif measure == 'euclidean':
            self.sim = euclidean_sim
        else:
            self.sim = cosine_sim

        self.max_violation = max_violation

    def forward(self, brand_ids, brand_emb, post_emb):

        # # brand升维情况下的相似度分数矩阵计算
        # post_emb = post_emb.view(post_emb.shape[0], post_emb.shape[1], 1)
        # # similarity matrix
        # scores = torch.empty((brand_emb.shape[0], brand_emb.shape[0]), device='cuda')
        # for i in range(post_emb.shape[0]):
        #     scores[i] = brand_emb.matmul(post_emb[i]).squeeze().mean(1)

        # brand不进行升维
        scores = self.sim(brand_emb, post_emb)

        # 秩约束
        # 计算每个正例在每个batch中的排名
        # 样本的权值与在batch中的排名有关
        # (数值矩阵 索引矩阵)
        # _, a11 = scores.sort(1, descending=True)
        # _, b11 = a11.sort(1)
        # rank_1 = (b11.diag() + 1).float()
        # rank_image = 1/(rank_1.shape[0] - rank_1 + 1) + 1
        # _, a22 = scores.sort(0, descending=True)
        # _, b22 = a22.sort(0)
        # rank_2 = (b22.diag() + 1).float()
        # rank_text = 1/(rank_2.shape[0] - rank_2 + 1) + 1
        diagonal = scores.diag().view(brand_emb.size(0), 1)
        # d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)
        # mask = torch.eye(scores.size(0)) > .5
        # I = mask
        # if torch.cuda.is_available():
        #     I = I.cuda()

        # cost_s = None
        # cost_im = None
        cost_b = None
        # cost_p = None
        # compare every diagonal score to scores in its column
        # if self.direction in ['i2t', 'all']:
        #     # caption retrieval
        #     cost_s = (self.margin + scores - d1).clamp(min=0)
        #     # 去除元素本身所在位置的影响
        #     cost_s = cost_s.masked_fill_(I, 0)
        # # compare every diagonal score to scores in its row
        # if self.direction in ['t2i', 'all']:
        #     # image retrieval
        #     cost_im = (self.margin + scores - d2).clamp(min=0)
        #     cost_im = cost_im.masked_fill_(I, 0)
        # 每行的值与对角线值的关系
        if self.direction in ['b2p', 'all']:
            cost_b = (self.margin + scores - d2).clamp(min=0)
            # cost = cost.masked_fill_(I, 0)
            # 将一个batch中同标签的样本带来的影响剔除
            mask = torch.zeros_like(scores)
            for i in range(mask.size(0)):
                for j in range(mask.size(1)):
                    if brand_ids[j] == brand_ids[i]:
                        mask[i][j] = 1
            mask = mask > 0.5
            if torch.cuda.is_available():
                mask = mask.cuda()
            cost_b = cost_b.masked_fill_(mask, 0)
        # # 每列的值与对角线值的关系
        # if self.direction in ['p2b', 'all']:
        #     cost_p = (self.margin + scores - d1).clamp(min=0)
        #     mask = torch.zeros_like(scores)
        #     for i in range(mask.size(0)):
        #         for j in range(mask.size(1)):
        #             if brand_ids[j] == brand_ids[i]:
        #                 mask[i][j] = 1
        #     mask = mask > 0.5
        #     mask = mask.cuda()
        #     cost_p = cost_p.masked_fill_(mask, 0)
        #
        if self.max_violation:
            if cost_b is not None:
                cost_b = cost_b.max(0)[0]
        #     if cost_p is not None:
        #         cost_p = cost_p.max(1)[0]
        #
        if cost_b is None:
            cost_b = torch.zeros(1)
            if torch.cuda.is_available():
                cost_b = cost_b.cuda()
        # if cost_p is None:
        #     cost_p = torch.zeros(1).cuda(
        #
        if self.cost_style == 'sum':
            total_loss = cost_b.sum()
        else:
            total_loss = cost_b.mean()
        return total_loss
        # keep the maximum violating negative for each queryasd
        # if self.max_violation:
        #     if cost_s is not None:
        #         cost_s = cost_s.max(1)[0]
        #         # print(len(cost_s))
        #     if cost_im is not None:
        #         cost_im = cost_im.max(0)[0]

        # cost_s = rank_image * cost_s
        # cost_im = rank_text * cost_im

        # # EET loss
        # M = scores.size(0)
        # # 约束同一个品牌到不同negative样本尽可能等距 e.g d(A,p_n1) = d(A,p_n2)
        # loss_ec1 = torch.tensor(0.0, device='cuda')
        # # 最大化类间距离
        # loss_m = torch.tensor(0.0, device='cuda')
        # value = []
        # total = []
        # # enumerate brand
        # for i in range(M):
        #     value.clear()
        #     # enumerate post
        #     for j in range(M):
        #         if brand_ids[j] != brand_ids[i]:
        #             value.append(scores[i][j])
        #             total.append(scores[i][j])
        #     tmp = torch.Tensor(value).cuda()
        #     loss_ec1 += tmp.var()
        #     loss_m += torch.sum(torch.exp(tmp))
        # loss_ec1 /= M
        # loss_m /= (M*(M-1))
        #
        # # 约束任意一对品牌和negative样本尽可能等距 如A~p_n1 B~p_n2之间
        # loss_e = torch.Tensor(total).var()/M
        #
        # loss_ec2 = loss_m + loss_e
        # # 最小化类内距离
        # loss_n = torch.exp(-scores.diag()).sum()/M
        # loss_eet = loss_ec1 + loss_ec2 + loss_n

        # if cost_s is None:
        #     cost_s = torch.zeros(1).cuda()
        # if cost_im is None:
        #     cost_im = torch.zeros(1).cuda()

        # if self.cost_style == 'sum':
        #     total_loss = cost_s.sum() + cost_im.sum()
        # else:
        #     total_loss = cost_s.mean() + cost_im.mean()

        # if self.loss_fun == 'eet':
        #     total_loss += loss_eet.cuda()

        # return total_loss
