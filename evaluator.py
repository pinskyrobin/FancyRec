# -*- coding: utf-8 -*
from __future__ import print_function

import time
import numpy as np
import torch

from util.constant import device
from util.util import AverageMeter
from util.ndcg import ndcg_at_k


# L2正则化
def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


# 余弦相似度
def cal_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    # l2规范化
    im = l2norm(im)
    s = l2norm(s)
    return im.mm(s.t())


# 对比实验random
def random_sim(num_brands, num_test_posts):
    return np.random.rand(num_brands, num_test_posts)


# 特征表示
def encode_data(model, data_loader, log_step=10, logging=print):
    """Encode all videos and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()

    model.brand_encoding.eval()
    if model.opt.single_modal_text:
        model.text_encoding.eval()
    elif model.opt.single_modal_visual:
        model.vid_encoding.eval()
    else:
        model.vid_encoding.eval()
        model.text_encoding.eval()
        model.fusion_encoding.eval()

    end = time.time()

    brands = torch.tensor([], dtype=torch.int).to(device)
    post_embs = torch.zeros((len(data_loader.dataset), model.opt.common_embedding_size)).to(device)

    with torch.no_grad():
        for i, (brand_ids, videos, captions, idxs, cap_ids, vid_ids) in enumerate(data_loader):

            brand_ids = brand_ids.to(device)
            brands = torch.cat((brands, brand_ids), 0)

            _, post_emb = model(brand_ids, videos, captions)

            if post_embs is None:
                post_embs = torch.zeros((len(data_loader.dataset), post_emb.size(1))).to(device)

            post_embs[np.array(idxs)] = post_emb

            batch_time.update(time.time() - end)
            end = time.time()

            if i % log_step == 0:
                logging('Process: [{0:2d}/{1:2d}]\t'
                        '{e_log}\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                            i, len(data_loader), batch_time=batch_time, e_log=str(model.logger)))
            del videos, captions

        return brands, post_embs


# 计算各个评价指标
def test_post_ranking(brand_num, metric, model, post_embs, brands):

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    aspect_model = model.brand_encoding.eval()

    brand_list = [i for i in range(brand_num)]
    brand_ = torch.LongTensor(brand_list).to(device)
    aspects = aspect_model(brand_)
    aspects = aspects.permute((1, 0, 2)).mean(0)

    scores = cal_sim(aspects, post_embs).data.cpu().numpy().copy()
    brands = brands.data.cpu().numpy().copy()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # scores = random_sim(aspects.shape[0], post_embs.shape[1])
    if metric == 'auc':
        queries = []
        ranks = np.zeros(scores.shape[0])

        for b in range(scores.shape[0]):
            predictions = [(scores[b, j], int(brands[j])) for j in range(scores.shape[1])]
            s_predictions = sorted(predictions, key=lambda x: x[0], reverse=True)

            pos = [v[0] for v in s_predictions if brand_list[b] == v[-1]]
            neg = [v[0] for v in s_predictions if brand_list[b] != v[-1]]
            sum = np.sum([len([el for el in neg if e > el]) for e in pos])

            if len(pos) != 0:
                rank_of_first_pos = list(zip(*s_predictions))[-1].index(brand_list[b])
                queries.append((rank_of_first_pos,
                                float(sum) / (len(pos) * len(neg)),
                                ndcg_at_k([1 if brand_list[b] == v[-1] else 0 for v in s_predictions], 10),
                                ndcg_at_k([1 if brand_list[b] == v[-1] else 0 for v in s_predictions], 50)))

                brands_tmp = np.array(brands)
                d = scores[b]
                inds = np.argsort(-d)
                brand_idx = brands_tmp[inds]
                rank = np.where(brand_idx == b)[0][0]
                ranks[b] = rank

        r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
        queries = list(zip(*queries))
        return (
            np.floor(np.median(queries[0])),  # MedR
            np.floor(np.mean(queries[0])),  # MeanR
            np.average(queries[1]),  # AUC
            # np.average(queries[2]),  # cAUC
            np.average(queries[2]),  # NDCG@10
            np.average(queries[3]),  # NDCG@50
            r1,  # recall@1
            r5,  # recall@5
            r10  # recall@10
        )
