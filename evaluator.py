# -*- coding: utf-8 -*
from __future__ import print_function

import numpy
import time
import numpy as np
from scipy.spatial import distance
import torch

from util.constant import device
from util.metric import getScorer
from util.util import AverageMeter, LogCollector
from util.ndcg import ndcg_at_k


# # l2正则化
# def l2norm(X):
#     """L2-normalize columns of X
#     """
#     norm = np.linalg.norm(X, axis=1, keepdims=True)
#     return 1.0 * X / norm


# # 计算品牌表示和posts表示的相似度分数
# def cal_sim(brands_emb, posts_emb, measure='cosine'):
#     result = None
#     if measure == 'cosine':
#         brands_emb = l2norm(brands_emb)
#         posts_emb = l2norm(posts_emb)
#         result = numpy.dot(brands_emb, posts_emb.T)
#     elif measure == 'euclidean':
#         result = distance.cdist(brands_emb, posts_emb, 'euclidean')
#     return result


# l2正则化
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


# 对提取的视频、图像、文本特征进一步编码(embedding)
def encode_data(model, data_loader, log_step=10, logging=print):
    """Encode all videos and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.vid_encoding.eval()
    model.text_encoding.eval()
    model.brand_encoding.eval()
    model.fusion_encoding.eval()

    end = time.time()

    # numpy array to keep all the embeddings
    # brands = []
    # post_embs = None
    brands = torch.tensor([], dtype=torch.int).to(device)
    post_embs = torch.zeros((len(data_loader.dataset), 1024)).to(device)
    # video_ids = [''] * len(data_loader.dataset)
    # caption_ids = [''] * len(data_loader.dataset)
    with torch.no_grad():
        for i, (brand_ids, videos, captions, idxs, cap_ids, vid_ids) in enumerate(data_loader):
            # make sure val logger is used
            model.logger = val_logger
            brand_ids = brand_ids.to(device)
            # brands.extend(brand_ids)
            brands = torch.cat((brands, brand_ids), 0)
            # compute the embeddings
            # 验证阶段不需要计算梯度
            _, post_emb = model(brand_ids, videos, captions)

            # initialize the numpy arrays given the size of the embeddings
            if post_embs is None:
                # brand_emb = np.zeros((len(data_loader.dataset), brand_emb.size(1)))
                post_embs = torch.zeros((len(data_loader.dataset), post_emb.size(1))).to(device)

            # preserve the embeddings by copying from gpu and converting to numpy
            post_embs[np.array(idxs)] = post_emb

            # brand_embs[idxs] = brand_emb.data.cpu().numpy().copy()
            # post_embs[np.array(idxs)] = post_emb.data.cpu().numpy().copy()

            # for j, idx in enumerate(idxs):
            #     caption_ids[idx] = cap_ids[j]
            #     video_ids[idx] = vid_ids[j]

            # measure elapsed time
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
    """
    :param metric:
    :param model:
    :param post_embs: all posts embeddings in test set
    :param brands: all (testing set)posts' brand information
    :return:
    """

    torch.cuda.empty_cache()
    aspect_model = model.brand_encoding.eval()
    # total brands are here
    brand_list = [i for i in range(brand_num)]
    brand_ = torch.LongTensor(brand_list).to(device)

    # aspect_model.to(device)

    # aspects = aspect_model(brand_)
    aspects = aspect_model(brand_)
    # brand_num*2048
    aspects = aspects.permute((1, 0, 2)).mean(0)
    # compute scores between brand and post
    # aspects = aspects.data.cpu().numpy().copy()
    # shape: brand_num * len(test_set)

    scores = cal_sim(aspects, post_embs).data.cpu().numpy().copy()
    brands = brands.data.cpu().numpy().copy()


    torch.cuda.empty_cache()
    # 对照实验1 随机相似度分数矩阵 # wmy
    # scores = random_sim(aspects.shape[0], post_embs.shape[1])
    if metric == 'auc':
        queries = []
        # compute r1, r5 and r10
        ranks = np.zeros(scores.shape[0])

        for b in range(scores.shape[0]):
            # print("processing brand_id {} in test set.".format(str(b)))
            # Computing evaluation metrics for a brand
            # tuple(test_post_code,score,brand_idx)
            predictions = [(scores[b, j], int(brands[j])) for j in range(scores.shape[1])]
            s_predictions = sorted(predictions, key=lambda x: x[0], reverse=True)

            pos = [v[0] for v in s_predictions if brand_list[b] == v[-1]]
            neg = [v[0] for v in s_predictions if brand_list[b] != v[-1]]
            # competitor in the same vertical
            # comp = [v[0] for v in s_predictions if (brand_list[b] != v[-1])]
            sum = np.sum([len([el for el in neg if e > el]) for e in pos])

            # pass the brand without data points
            if len(pos) != 0:
                # first positive sample's position
                rank_of_first_pos = list(zip(*s_predictions))[-1].index(brand_list[b])
                queries.append((rank_of_first_pos,
                                float(sum) / (len(pos) * len(neg)),
                                # float(np.sum([len([el for el in comp if e > el])for e in pos]))/(len(pos) * len(
                                # comp)+1),
                                ndcg_at_k([1 if brand_list[b] == v[-1] else 0 for v in s_predictions], 10),
                                ndcg_at_k([1 if brand_list[b] == v[-1] else 0 for v in s_predictions], 50)))

                # print("brands", brands)
                brands_tmp = np.array(brands)
                # print("brands_tmp",brands_tmp)
                # exit(0)
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


# recall@k, Med r, Mean r for Text-to-Video Retrieval
def t2i(c2i, n_caption=5):
    """
    Text->Videos (Text-to-Video Retrieval)
    c2i: (5N, N) matrix of caption to video_frames errors
    vis_details: if true, return a dictionary for ROC visualization purposes
    """
    # print("errors matrix shape: ", c2i.shape)
    # (59800,2990)
    assert c2i.shape[0] / c2i.shape[1] == n_caption, c2i.shape
    ranks = np.zeros(c2i.shape[0])
    # print(ranks)

    for i in range(len(ranks)):
        d_i = c2i[i]
        # print(d_i)

        inds = np.argsort(d_i)
        # print(inds)
        rank = np.where(inds == i / n_caption)[0][0]
        ranks[i] = rank

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return map(float, [r1, r5, r10, medr, meanr])


# recall@k, Med r, Mean r for Video-to-Text Retrieval
def i2t(c2i, n_caption=5):
    """
    Videos->Text (Video-to-Text Retrieval)
    c2i: (5N, N) matrix of caption to video_frames errors
    """
    # remove duplicate videos
    # print("errors matrix shape: ", c2i.shape)
    assert c2i.shape[0] / c2i.shape[1] == n_caption, c2i.shape
    ranks = np.zeros(c2i.shape[1])

    for i in range(len(ranks)):
        d_i = c2i[:, i]
        inds = np.argsort(d_i)

        rank = np.where(inds / n_caption == i)[0][0]
        ranks[i] = rank

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    return map(float, [r1, r5, r10, medr, meanr])


# mAP for Text-to-Video Retrieval
def t2i_map(c2i, n_caption=5):
    """
    Text->Videos (Text-to-Video Retrieval)
    c2i: (5N, N) matrix of caption to video_frames errors
    """
    # print("errors matrix shape: ", c2i.shape)
    assert c2i.shape[0] / c2i.shape[1] == n_caption, c2i.shape

    scorer = getScorer('AP')
    perf_list = []
    for i in range(c2i.shape[0]):
        d_i = c2i[i, :]
        labels = [0] * len(d_i)
        labels[i // n_caption] = 1

        sorted_labels = [labels[x] for x in np.argsort(d_i)]
        current_score = scorer.score(sorted_labels)
        perf_list.append(current_score)

    return np.mean(perf_list)


# mAP for Video-to-Text Retrieval
def i2t_map(c2i, n_caption=5):
    """
    Videos->Text (Video-to-Text Retrieval)
    c2i: (5N, N) matrix of caption to video_frames errors
    """
    # print("errors matrix shape: ", c2i.shape)
    assert c2i.shape[0] / c2i.shape[1] == n_caption, c2i.shape

    scorer = getScorer('AP')
    perf_list = []
    for i in range(c2i.shape[1]):
        d_i = c2i[:, i]
        labels = [0] * len(d_i)
        labels[i * n_caption:(i + 1) * n_caption] = [1] * n_caption

        sorted_labels = [labels[x] for x in np.argsort(d_i)]
        current_score = scorer.score(sorted_labels)
        perf_list.append(current_score)

    return np.mean(perf_list)


def t2i_inv_rank(c2i, n_caption=1):
    """
    Text->Videos (Text-to-Video Retrieval)
    c2i: (5N, N) matrix of caption to video_frames errors
    n_caption: number of captions of each image/video_frames
    """
    assert c2i.shape[0] / c2i.shape[1] == n_caption, c2i.shape
    inv_ranks = np.zeros(c2i.shape[0])

    for i in range(len(inv_ranks)):
        d_i = c2i[i, :]
        inds = np.argsort(d_i)

        rank = np.where(inds == i / n_caption)[0]
        inv_ranks[i] = sum(1.0 / (rank + 1))

    return np.mean(inv_ranks)


def i2t_inv_rank(c2i, n_caption=1):
    """
    Videos->Text (Video-to-Text Retrieval)
    c2i: (5N, N) matrix of caption to video_frames errors
    n_caption: number of captions of each image/video_frames
    """
    assert c2i.shape[0] / c2i.shape[1] == n_caption, c2i.shape
    inv_ranks = np.zeros(c2i.shape[1])

    for i in range(len(inv_ranks)):
        d_i = c2i[:, i]
        inds = np.argsort(d_i)

        rank = np.where(inds / n_caption == i)[0]
        inv_ranks[i] = sum(1.0 / (rank + 1))

    return np.mean(inv_ranks)


def i2t_inv_rank_multi(c2i, n_caption=2):
    """
    Text->videos (Image Search)
    c2i: (5N, N) matrix of caption to image errors
    n_caption: number of captions of each image/video_frames
    """
    # print("errors matrix shape: ", c2i.shape)
    assert c2i.shape[0] / c2i.shape[1] == n_caption, c2i.shape
    inv_ranks = np.zeros(c2i.shape[1])

    result = []
    for i in range(n_caption):
        idx = range(i, c2i.shape[0], n_caption)
        sub_c2i = c2i[idx, :]
        score = i2t_inv_rank(sub_c2i, n_caption=1)
        result.append(score)
    return result
