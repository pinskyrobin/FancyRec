from torch import nn
import torch
import torch.nn.functional as F
import numpy as np

from util.constant import device


# def cosine_sim(emb1, emb2):
#     """compute cosine similarity of two embeddings
#     Args:
#         emb1
#         emb2
#     """
#     return emb1.mm(emb2.t())

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


class MaxMargin_coot(nn.Module):
    """Regular Contrastive Loss between 2 groups of embeddings
    inputs shape (batch, embed_dim)
    Ref: COOT: Cooperative Hierarchical Transformer for Video-Text Representation Learning, NeurIPS 2020
    """

    def __init__(self, use_cuda: bool, margin: float = 0.1):
        super(MaxMargin_coot, self).__init__()
        self.margin = margin
        self.sim = cosine_sim
        self.use_cuda = use_cuda

    def forward(self, im, s):
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)
        cost_s = (self.margin + scores - d1).clamp(min=0)
        cost_im = (self.margin + scores - d2).clamp(min=0)
        mask = torch.eye(scores.size(0)) > .5
        if self.use_cuda:
            mask = mask.to(device)
        cost_s = cost_s.masked_fill_(mask, 0)
        cost_im = cost_im.masked_fill_(mask, 0)
        return (cost_s.sum() + cost_im.sum()).div(im.shape[0] * s.shape[0])


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


class CrossCLR_noq(nn.Module):
    """
    CrossCLR Loss between 2 groups of embeddings
    """

    def __init__(self, temperature=0.03, temperature_weights=0.0035, negative_weight=0.8,
                 score_threshold=0.7, logger=None):
        super(CrossCLR_noq, self).__init__()
        self.logit_scale = nn.Parameter(torch.ones([]))
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')  # torch.nn.CrossEntropyLoss()
        self.temperature = temperature  #
        self.logger = logger

        self.score_threshold = score_threshold
        self.temp_w = temperature_weights  # Temperature for scaling weights.
        self.negative_w = negative_weight  # Weight of negative scores.
        self.logger.info("===" * 30)
        self.logger.info("Temp:{}, TempW:{}, NegW:{}, Sthrsh:{}".format(self.temperature, self.temp_w,
                                                                        self.negative_w, self.score_threshold))
        self.logger.info("===" * 30)
        # create the queue

    def compute_loss(self, logits, mask):

        loss = - torch.log((F.softmax(logits, dim=1) * mask).sum(1))
        return loss  # loss.mean()

    def _get_positive_mask(self, batch_size):
        diag = np.eye(batch_size)
        mask = torch.from_numpy(diag)
        mask = (1 - mask)
        return mask.to(device)

    def _get_positive_mask_bank(self, k, batch_size, ptr):
        diag = np.eye(batch_size)
        mask = torch.from_numpy(diag)
        # mask = (1 - mask)

        diag_bank = np.ones((batch_size, k))
        mask_bank = torch.from_numpy(diag_bank)

        if (ptr + batch_size) > k:
            inp_feat_k = batch_size - (ptr + batch_size - k)
            mask_bank[:, ptr:] -= mask[:, :inp_feat_k]
        else:
            mask_bank[:, ptr:ptr + batch_size] -= mask

        return mask_bank.to(device)

    def forward(self, brand, post, input_brand=None, input_post=None):
        """
        Inputs shape (batch, embed_dim)

        Args:
            brand: Brand embeddings (batch, embed_dim)
            post: Post embeddings (batch, embed_dim)
            input_post:
            input_brand:
        Returns:
        """

        brand = nn.functional.normalize(brand, dim=1)
        post = nn.functional.normalize(post, dim=1)
        input_brand = input_brand.to(device)
        input_post = input_post.to(device)

        logits_per_image = brand @ post.t()
        logits_per_text = post @ brand.t()

        logits_clstr_brand = brand @ brand.t()
        logits_clstr_post = post @ post.t()

        logits_per_image /= self.temperature
        logits_per_text /= self.temperature
        logits_clstr_brand /= self.temperature
        logits_clstr_post /= self.temperature

        positive_mask = self._get_positive_mask(brand.shape[0])
        sim_scores_brand = (input_brand @ input_brand.t()) * positive_mask
        sim_scores_post = (input_post @ input_post.t()) * positive_mask

        avg_sim_brand = torch.mean(sim_scores_brand, dim=1)
        avg_sim_post = torch.mean(sim_scores_post, dim=1)

        sorted_brand, indices_brand = torch.sort(avg_sim_brand)
        sorted_post, indices_post = torch.sort(avg_sim_post)
        sorted_brand = sorted_brand / sorted_brand.max(dim=-1, keepdim=True)[0]
        sorted_post = sorted_post / sorted_post.max(dim=-1, keepdim=True)[0]

        # ======================================================
        # Find index of influential samples and remove them from negative set
        indices_brand_thresh = indices_brand[sorted_brand < self.score_threshold]
        indices_post_thresh = indices_post[sorted_post < self.score_threshold]

        logits_clstr_brand = logits_clstr_brand * positive_mask
        logits_clstr_post = logits_clstr_post * positive_mask

        negatives_brand = logits_clstr_brand[:, indices_brand_thresh]
        negatives_post = logits_clstr_post[:, indices_post_thresh]

        batch_size = input_brand.shape[0]

        brand_logits_prune = logits_per_image
        post_logits_prune = logits_per_text

        prune_pos = 1
        if prune_pos:
            sorted_brand2, indices_brand2 = torch.sort(avg_sim_brand)
            sorted_post2, indices_post2 = torch.sort(avg_sim_post)
            sorted_brand2 = sorted_brand2 / sorted_brand2.max(dim=-1, keepdim=True)[0]
            sorted_post2 = sorted_post2 / sorted_post2.max(dim=-1, keepdim=True)[0]
            indices_brand_thresh2 = indices_brand2[sorted_brand2 > self.score_threshold]
            indices_post_thresh2 = indices_post2[sorted_post2 > self.score_threshold]

            mask_prune_pos_brand = torch.ones_like(logits_per_image)
            mask_prune_pos_post = torch.ones_like(logits_per_text)

            mask_prune_pos_brand[:, indices_brand_thresh2] = 0
            mask_prune_pos_post[:, indices_post_thresh2] = 0

            for i in range(batch_size):
                if mask_prune_pos_brand[i, i] == 0:
                    mask_prune_pos_brand[i, i] = 1
                if mask_prune_pos_post[i, i] == 0:
                    mask_prune_pos_post[i, i] = 1

            brand_logits_prune = logits_per_image * mask_prune_pos_brand
            post_logits_prune = logits_per_text * mask_prune_pos_post

        brand_logits = torch.cat([brand_logits_prune, self.negative_w * negatives_brand], dim=1)
        post_logits = torch.cat([post_logits_prune, self.negative_w * negatives_post], dim=1)

        diag = np.eye(batch_size)
        mask_brand = torch.from_numpy(diag).to(device)
        mask_post = torch.from_numpy(diag).to(device)

        multi_pos = 0
        num_p = 5
        mp_score = 0.15
        if multi_pos:
            positive_mask = self._get_positive_mask(brand.shape[0])
            sim_mask_brand = (input_brand @ input_brand.t()) * positive_mask
            sim_mask_post = (input_post @ input_post.t()) * positive_mask
            _, topk_idx_brand = torch.topk(sim_mask_brand, num_p, dim=1)
            topk_onehot_brand = torch.zeros_like(sim_mask_brand)
            topk_onehot_brand.scatter_(1, topk_idx_brand, 1)
            mask_brand[topk_onehot_brand.bool()] = mp_score

            _, topk_idx_post = torch.topk(sim_mask_post, num_p, dim=1)
            topk_onehot_post = torch.zeros_like(sim_mask_post)
            topk_onehot_post.scatter_(1, topk_idx_post, 1)
            mask_post[topk_onehot_post.bool()] = mp_score

        mask_neg_brand = torch.zeros_like(negatives_brand)
        mask_neg_post = torch.zeros_like(negatives_post)
        mask_brand = torch.cat([mask_brand, mask_neg_brand], dim=1)
        mask_post = torch.cat([mask_post, mask_neg_post], dim=1)

        loss_brand = self.compute_loss(brand_logits, mask_brand)
        loss_post = self.compute_loss(post_logits, mask_post)

        w_brand = (avg_sim_brand / sum(avg_sim_brand))
        w_post = (avg_sim_post / sum(avg_sim_post))
        loss_brand = loss_brand * torch.exp(w_brand / self.temp_w)
        loss_post = loss_post * torch.exp(w_post / self.temp_w)

        loss_brand = sum(loss_brand) / (sum(torch.exp(w_brand / self.temp_w)))
        loss_post = sum(loss_post) / (sum(torch.exp(w_post / self.temp_w)))

        loss = (loss_brand + loss_post) / 2

        return loss


class ContrastiveLoss(nn.Module):

    def __init__(self, opt, K=4096, temperature=0.03, weight_temperature=35e4, negative_weight=0.8,
                 score_threshold=0.99, logger=None, cost_style='sum'):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.K = K
        self.logit_scale = nn.Parameter(torch.ones([]))
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.temperature = temperature
        self.weight_temperature = weight_temperature
        self.score_threshold = score_threshold
        self.logger = logger
        self.cost_style = cost_style
        self.negative_w = negative_weight  # Weight of negative samples logits.

        # create the queue
        self.register_buffer("queue", torch.zeros(K, self.opt.common_embedding_size))
        # self.queue = nn.functional.normalize(self.queue, dim=0)
        # self.queue = torch.zeros(K, self.opt.common_embedding_size)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, post):
        batch_size = post.shape[0]

        ptr = int(self.queue_ptr)
        # assert self.K % batch_size == 0 for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr: ptr + batch_size] = post
        prev_ptr = ptr

        self.queue_ptr[0] = (ptr + batch_size) % self.K  # move pointer

        return prev_ptr

    def _min_max_norm(self, tensor):
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)
        return (tensor - min_val) / (max_val - min_val)

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
        return loss.sum()  # loss.mean()

    def forward(self, brand, post):
        # brand = nn.functional.normalize(brand, dim=1)
        # post = nn.functional.normalize(post, dim=1)
        # all_post = torch.cat([post, self.queue], dim=0)
        #
        # intra_pos_mask = self._get_positive_mask(all_post.shape[0])
        # inter_modality_logits = brand @ post.t() / self.temperature
        # intra_modality_logits = all_post @ all_post.t() * intra_pos_mask / self.temperature
        #
        # curr_post_logits = torch.sum(intra_modality_logits[: post.shape[0], : post.shape[0]].clone(), dim=1)
        # curr_sorted_post, curr_sorted_post_index = torch.sort(curr_post_logits)
        # curr_sorted_post = self._min_max_norm(curr_sorted_post)
        #
        # influential_post_index = curr_sorted_post_index[curr_sorted_post >= self.score_threshold]
        # mask_influential_post = self._get_negative_mask(post, influential_post_index)
        #
        # update_queue = post * mask_influential_post
        # prev_ptr = self._dequeue_and_enqueue(update_queue)
        #
        # queue_pos_mask = self._get_positive_mask(self.K)
        # queue_logits = self.queue @ self.queue.t() * queue_pos_mask
        # connection = torch.mean(queue_logits, dim=1)[prev_ptr: prev_ptr + post.shape[0]].clone()
        # weight = torch.exp(connection / self.weight_temperature)
        #
        # logits = self._crossmodal_softmax(inter_modality_logits,
        #                                   queue_logits[prev_ptr: prev_ptr + post.shape[0]].clone() / self.temperature,
        #                                   self.negative_w)
        #
        # loss = self._compute_loss(logits, weight)

        brand = nn.functional.normalize(brand, dim=1)
        post = nn.functional.normalize(post, dim=1)

        prev_ptr = self._dequeue_and_enqueue(post)

        ori_logits = post @ self.queue.t()
        queue_pos_mask = self._get_positive_mask(ori_logits)
        inter_modality_logits = brand @ post.t() / self.temperature
        intra_modality_logits = ori_logits * queue_pos_mask / self.temperature

        # post_logits = intra_modality_logits[prev_ptr: prev_ptr + post.shape[0]]

        # sum_post_logits = torch.sum(post_logits, dim=1)
        # sorted_post, sorted_post_index = torch.sort(sum_post_logits)
        # sorted_post = self._min_max_norm(sorted_post)

        # influential_post_index = sorted_post_index[sorted_post >= self.score_threshold]
        # mask_influential_post = self._get_negative_mask(post_logits, influential_post_index)

        connection = torch.mean(intra_modality_logits, dim=1)
        weight = torch.exp(connection / self.weight_temperature)

        logits = self._crossmodal_softmax(inter_modality_logits, intra_modality_logits, self.negative_w)
        loss = self._compute_loss(logits, weight)

        return loss
