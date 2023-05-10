# -*- coding: utf-8 -*
import torch
import torch.nn as nn
import numpy as np

from util.constant import device

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


# 实验测试用loss，仅设计品牌隐向量
class LabLoss(nn.Module):
    def __init__(self):
        super(LabLoss, self).__init__()

    def forward(self, brand_embs):
        s = cosine_sim(brand_embs, brand_embs)
        # print(s)
        mask = torch.eye(s.size(0)) > .5
        I = mask.to(device)
        s.masked_fill_(I, 0)
        # print(s)
        loss_lab = (torch.sum(torch.exp(s)) - s.size(0)).to(device)
        return loss_lab / s.size(0)


# 传统的三元组损失（带排序权重）
class TripletLoss(nn.Module):
    """
    triplet ranking loss
    """

    def __init__(self, margin=0, measure='cosine', max_violation=False, cost_style='sum', direction='all', loss_fun='mrl'):
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

        post_emb = post_emb.view(post_emb.shape[0], post_emb.shape[1], 1)
        # similarity matrix
        scores = torch.empty((brand_emb.shape[0], brand_emb.shape[0])).to(device)
        for i in range(post_emb.shape[0]):
            scores[i] = brand_emb.matmul(post_emb[i]).squeeze()

        # return: (values, indices)
        _, a11 = scores.sort(1, descending=True)
        # bii[i][j] = aii[i][a11[i][j]]
        _, b11 = a11.sort(1)
        rank_1 = (b11.diag() + 1).float()
        rank_p = 1 / (rank_1.shape[0] - rank_1 + 1) + 1

        _, a22 = scores.sort(0, descending=True)
        _, b22 = a22.sort(0)
        rank_2 = (b22.diag() + 1).float()
        rank_b = 1 / (rank_2.shape[0] - rank_2 + 1) + 1

        diagonal = scores.diag().view(brand_emb.size(0), 1)

        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        cost_b = None
        cost_p = None

        # omit the same brand
        mask = torch.from_numpy(np.fromiter((brand_ids[i] == brand_ids[j]
                                             for i in range(len(brand_ids))
                                             for j in range(len(brand_ids))), bool)) \
            .view(len(brand_ids), len(brand_ids)).to(device)

        # 每列的值与对角线值的关系
        if self.direction in ['p2b', 'all']:
            cost_p = (self.margin + scores - d1).clamp(min=0)
            cost_p = cost_p.masked_fill_(mask, 0)

        # 每行的值与对角线值的关系
        if self.direction in ['b2p', 'all']:
            cost_b = (self.margin + scores - d2).clamp(min=0)
            cost_b = cost_b.masked_fill_(mask, 0)

        cost_p = rank_p * cost_p
        cost_b = rank_b * cost_b

        if cost_b is None:
            cost_b = torch.zeros(1).to(device)
        if cost_p is None:
            cost_p = torch.zeros(1).to(device)

        if self.cost_style == 'sum':
            total_loss = cost_b.sum() + cost_p.sum()
        else:
            total_loss = cost_b.mean() + cost_p.mean()
        return total_loss
