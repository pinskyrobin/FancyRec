from torch import nn
import torch
import torch.nn.functional as F
import numpy as np

from util.constant import device


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


class CrossCLR_onlyIntraModality(nn.Module):
    """
    CrossCLR Loss between 2 groups of embeddings - Only Intra Modality alignment
    ICCV 2021
    """

    def __init__(self, temperature=0.03, negative_weight=0.8, logger=None, cost_style='sum'):
        super(CrossCLR_onlyIntraModality, self).__init__()
        self.logit_scale = nn.Parameter(torch.ones([]))
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.temperature = temperature
        self.logger = logger
        self.cost_style = cost_style
        self.negative_w = negative_weight  # Weight of negative samples logits.

    def compute_loss(self, logits, mask):
        return - torch.log((F.softmax(logits, dim=1) * mask).sum(1))

    def _get_positive_mask(self, batch_size):
        diag = np.eye(batch_size, dtype=np.float32)
        mask = torch.from_numpy(diag)
        mask = (1 - mask)
        return mask.to(device)

    def forward(self, brand, post):
        """
        Inputs shape (batch, embed_dim)
        Args:
            brand: Brand embeddings (batch, embed_dim)
            post: Post embeddings (batch, embed_dim)
        Returns:
        """

        # Calculate rank
        _post = post.view(post.shape[0], post.shape[1], 1)
        scores = torch.empty((brand.shape[0], brand.shape[0])).to(device)
        for i in range(_post.shape[0]):
            scores[i] = brand.matmul(_post[i]).squeeze()

        # Return: (values, indices)
        _, a11 = scores.sort(1, descending=True)
        # bii[i][j] = aii[i][a11[i][j]]
        _, b11 = a11.sort(1)
        rank_1 = (b11.diag() + 1).float()
        rank_p = 1 / (rank_1.shape[0] - rank_1 + 1) + 1

        _, a22 = scores.sort(0, descending=True)
        _, b22 = a22.sort(0)
        rank_2 = (b22.diag() + 1).float()
        rank_b = 1 / (rank_2.shape[0] - rank_2 + 1) + 1

        batch_size = brand.shape[0]

        # Normalize features
        brand = nn.functional.normalize(brand, dim=1)
        post = nn.functional.normalize(post, dim=1)

        # Inter-modality alignment
        logits_per_brand = brand @ post.t()
        logits_per_post = post @ brand.t()

        # Intra-modality alignment
        logits_clstr_brand = brand @ brand.t()
        logits_clstr_post = post @ post.t()

        logits_per_brand /= self.temperature
        logits_per_post /= self.temperature
        logits_clstr_brand /= self.temperature
        logits_clstr_post /= self.temperature

        positive_mask = self._get_positive_mask(brand.shape[0])
        negatives_brand = logits_clstr_brand * positive_mask
        negatives_post = logits_clstr_post * positive_mask

        brand_logits = torch.cat([logits_per_brand, self.negative_w * negatives_brand], dim=1)
        post_logits = torch.cat([logits_per_post, self.negative_w * negatives_post], dim=1)

        diag = np.eye(batch_size, dtype=np.float32)
        mask_brand = torch.from_numpy(diag).to(device)
        mask_post = torch.from_numpy(diag).to(device)

        mask_neg_b = torch.zeros_like(negatives_brand)
        mask_neg_p = torch.zeros_like(negatives_post)
        mask_b = torch.cat([mask_brand, mask_neg_b], dim=1)
        mask_p = torch.cat([mask_post, mask_neg_p], dim=1)

        loss_b = rank_b * self.compute_loss(brand_logits, mask_b)
        loss_p = rank_p * self.compute_loss(post_logits, mask_p)

        return (loss_b.sum() + loss_p.sum()) / 2 if self.cost_style == 'sum' else (loss_b.mean() + loss_p.mean()) / 2


class ContrastiveLoss(nn.Module):

    def __init__(self, opt, temperature=0.03, negative_weight=0.8):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.logit_scale = nn.Parameter(torch.ones([]))
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.temperature = temperature
        self.cost_style = opt.cost_style
        self.negative_w = negative_weight  # Weight of negative samples logits.

        # create the queue
        self.register_buffer("queue", torch.zeros(opt.queue_size, opt.common_embedding_size))
        # self.queue = nn.functional.normalize(self.queue, dim=0)
        # self.queue = torch.zeros(K, self.opt.common_embedding_size)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, post):
        batch_size = post.shape[0]

        ptr = int(self.queue_ptr)
        # assert self.opt.queue_size % batch_size == 0 for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr: ptr + batch_size] = post
        self.queue_ptr[0] = (ptr + batch_size) % self.opt.queue_size  # move pointer

    def _get_positive_mask(self, tensor):
        # diag = np.eye(batch_size)
        # mask = torch.from_numpy(diag)
        # mask = (1 - mask)

        mask = torch.ones_like(tensor)
        ptr = self.queue_ptr[0]
        for i in range(tensor.shape[0]):
            mask[i][ptr] = 0
            ptr = ptr + 1
        return mask.to(device)

    def _get_negative_mask(self, tensor, index):
        mask = torch.ones_like(tensor)
        mask[index] = 0
        return mask.to(device)

    def _crossmodal_softmax(self, inter_logits, intra_logits, weight):
        exp_inter_logits = torch.exp(inter_logits)
        exp_intra_logits = torch.exp(intra_logits)
        exp_sum = exp_inter_logits.sum(dim=1) + weight * exp_intra_logits.sum(dim=1)
        return torch.diag(exp_inter_logits).t() / exp_sum

    def _compute_loss(self, logits, weight):
        loss = - torch.log(logits) * weight
        if self.cost_style == 'sum':
            return loss.sum()  # loss.mean()
        elif self.cost_style == 'mean':
            return loss.mean()

    def forward(self, brand, post):

        # Calculate rank
        _post = post.view(post.shape[0], post.shape[1], 1)
        scores = torch.empty((brand.shape[0], brand.shape[0])).to(device)
        for i in range(_post.shape[0]):
            scores[i] = brand.matmul(_post[i]).squeeze()

        # Return: (values, indices)
        _, a = scores.sort(1, descending=True)
        # bii[i][j] = aii[i][a11[i][j]]
        _, b = a.sort(1)
        rank_1 = (b.diag() + 1).float()
        weight = 1 / (rank_1.shape[0] - rank_1 + 1) + 1

        brand = nn.functional.normalize(brand, dim=1)
        post = nn.functional.normalize(post, dim=1)

        if self.opt.no_queue or self.opt.no_intra:
            ori_logits = post @ post.t()
            queue_pos_mask = self._get_positive_mask(ori_logits)
        else:
            self._dequeue_and_enqueue(post)
            ori_logits = post @ self.queue.t()
            queue_pos_mask = self._get_positive_mask(ori_logits)
        inter_modality_logits = brand @ post.t() / self.temperature
        intra_modality_logits = ori_logits * queue_pos_mask / self.temperature

        if self.opt.no_intra:
            zero_intra = torch.zeros_like(intra_modality_logits)
            logits = self._crossmodal_softmax(inter_modality_logits, zero_intra, self.negative_w)
        else:
            logits = self._crossmodal_softmax(inter_modality_logits, intra_modality_logits, self.negative_w)
        loss = self._compute_loss(logits, weight)

        return loss
