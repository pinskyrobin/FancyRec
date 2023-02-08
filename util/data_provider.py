# -*- coding: utf-8 -*-
import json

import torch
import torch.utils.data as data
import numpy as np
from transformers import BertTokenizer
from basic.util import get_visual_id, read_dict
from util.vocab import clean_str
import os
VIDEO_MAX_LEN = 64
# global
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def info(opt):
    img_info = read_dict(os.path.join(opt.rootpath,"img_info.txt"))
    cls_file = os.path.join(opt.rootpath, "cls.txt")
    cls_file = open(cls_file, 'r').read()
    cls_info = json.loads(cls_file)
    return img_info, cls_info


def collate_frame_transformers_fn(data):
    """build mini-batch from a list of (video_frames, caption) tuples
        process text features with transformers.
    """
    if data[0][2] is not None:
        data.sort(key=lambda x: len(x[2]), reverse=True)
    brand_ids, videos, captions, cap_bows, idxs, cap_ids, video_ids = zip(*data)
    # print("one batch of captions:", captions)
    video_lengths = [min(VIDEO_MAX_LEN, len(frame)) for frame in videos]
    frame_vec_len = len(videos[0][0])
    vidoes = torch.zeros(len(videos), max(video_lengths), frame_vec_len)
    vidoes_origin = torch.zeros(len(videos), frame_vec_len)
    vidoes_mask = torch.zeros(len(videos), max(video_lengths))
    for i, frames in enumerate(videos):
        end = video_lengths[i]
        vidoes[i, :end, :] = frames[:end, :]
        vidoes_origin[i, :] = torch.mean(frames, 0)
        vidoes_mask[i, :end] = 1.0

    if captions[0] is not None:
        text = tokenizer(captions, padding=True, truncation=True, return_tensors="pt")
        tokens = text['input_ids']
        # print(tokens.shape)
        # print(tokens.device)
        type_ids = text['token_type_ids']
        # print(type_ids.shape)
        masks = text['attention_mask']
        # print(masks.shape)
    else:
        tokens = None
        type_ids = None
        masks = None

    cap_bows = torch.stack(cap_bows, 0) if cap_bows[0] is not None else None
    video_data = (vidoes, vidoes_origin, video_lengths, vidoes_mask)
    text_data = (cap_bows, tokens, type_ids, masks)
    brand_ids = torch.LongTensor(brand_ids)

    return brand_ids, video_data, text_data, idxs, cap_ids, video_ids


def collate_frame_gru_fn(data):
    """
    Build mini-batch tensors from a list of (video_frames, caption) tuples.
    """
    # Sort a data list by caption length
    if data[0][2] is not None:
        data.sort(key=lambda x: len(x[2]), reverse=True)
    # example  change [(1,2,3,4,5),(6,7,8,9,10)]  ==> [(1,6),(2,7),(3,8),(4,9),(5,10)]
    # 根据数据集的构造方式可知 这里videos可能是图像，也可能是视频
    # 每个batch中视觉特征既有图像也有视频
    brand_ids, videos, captions, cap_bows, idxs, cap_ids, video_ids = zip(*data)

    # Merge videos (convert tuple of 1D tensor to 4D tensor)
    # 每个视频的帧数 最多64
    video_lengths = [min(VIDEO_MAX_LEN, len(frame)) for frame in videos]
    # 2048
    frame_vec_len = len(videos[0][0])
    # 按一个batch中帧数最多的视频为基准  其他帧数不足的填充0  batch_size*frame_num*feature_dim
    vidoes = torch.zeros(len(videos), max(video_lengths), frame_vec_len)
    # 每个视频的平均特征
    vidoes_origin = torch.zeros(len(videos), frame_vec_len)
    # 记录每个视频的有效帧数
    vidoes_mask = torch.zeros(len(videos), max(video_lengths))
    for i, frames in enumerate(videos):
        end = video_lengths[i]
        vidoes[i, :end, :] = frames[:end, :]
        vidoes_origin[i, :] = torch.mean(frames, 0)
        vidoes_mask[i, :end] = 1.0

    if captions[0] is not None:
        # Merge captions (convert tuple of 1D tensor to 2D tensor)
        lengths = [len(cap) for cap in captions]
        # 视频的句子描述同理  按batch中最长的句子为标准 其他短句填充0
        target = torch.zeros(len(captions), max(lengths)).long()
        # 记录有效句子长度
        words_mask = torch.zeros(len(captions), max(lengths))
        for i, cap in enumerate(captions):
            end = lengths[i]
            target[i, :end] = cap[:end]
            words_mask[i, :end] = 1.0
    else:
        target = None
        lengths = None
        words_mask = None

    cap_bows = torch.stack(cap_bows, 0) if cap_bows[0] is not None else None

    video_data = (vidoes, vidoes_origin, video_lengths, vidoes_mask)
    text_data = (target, cap_bows, lengths, words_mask)
    brand_ids = torch.LongTensor(brand_ids)

    return brand_ids, video_data, text_data, idxs, cap_ids, video_ids


def collate_frame(data):
    videos, idxs, video_ids = zip(*data)

    # Merge videos (convert tuple of 1D tensor to 4D tensor)

    video_lengths = [min(VIDEO_MAX_LEN, len(frame)) for frame in videos]
    frame_vec_len = len(videos[0][0])
    vidoes = torch.zeros(len(videos), max(video_lengths), frame_vec_len)
    videos_origin = torch.zeros(len(videos), frame_vec_len)
    vidoes_mask = torch.zeros(len(videos), max(video_lengths))
    for i, frames in enumerate(videos):
        end = video_lengths[i]
        vidoes[i, :end, :] = frames[:end, :]
        videos_origin[i, :] = torch.mean(frames, 0)
        vidoes_mask[i, :end] = 1.0

    video_data = (vidoes, videos_origin, video_lengths, vidoes_mask)

    return video_data, idxs, video_ids


def collate_text(data):
    if data[0][0] is not None:
        data.sort(key=lambda x: len(x[0]), reverse=True)
    captions, cap_bows, idxs, cap_ids = zip(*data)

    if captions[0] is not None:
        # Merge captions (convert tuple of 1D tensor to 2D tensor)
        lengths = [len(cap) for cap in captions]
        target = torch.zeros(len(captions), max(lengths)).long()
        words_mask = torch.zeros(len(captions), max(lengths))
        for i, cap in enumerate(captions):
            end = lengths[i]
            target[i, :end] = cap[:end]
            words_mask[i, :end] = 1.0
    else:
        target = None
        lengths = None
        words_mask = None

    cap_bows = torch.stack(cap_bows, 0) if cap_bows[0] is not None else None

    text_data = (target, cap_bows, lengths, words_mask)

    return text_data, idxs, cap_ids



class Dataset4DualEncoding(data.Dataset):
    """
    Load captions and video_frames frame features by pre-trained CNN model.
    """

    def __init__(self, opt, cap_file, video_feat, img_feat, bow2vec, vocab, text_net, n_caption=None, video2frames=None):
        # Captions
        self.captions = {}
        self.cap_ids = []
        self.video_ids = set()
        self.video2frames = video2frames
        self.text_net = text_net
        self.opt = opt

        with open(cap_file) as cap_reader:
            for line in cap_reader.readlines():
                # get caption_id and sentence
                # print(line)
                try:
                    cap_id, caption = line.strip().split(" ", 1)
                except:
                    continue
                video_id = get_visual_id(cap_id)
                self.captions[cap_id] = caption
                self.cap_ids.append(cap_id)
                self.video_ids.add(video_id)
        # 视频特征
        self.video_feat = video_feat
        # 图像特征
        self.img_feat = img_feat
        # 图像信息与品牌信息
        self.img_info, self.brand_info = info(self.opt)
        self.bow2vec = bow2vec
        self.vocab = vocab
        self.length = len(self.cap_ids)
        # print("total samples ", len(self.cap_ids))
        if n_caption is not None:
            assert len(self.video_ids) * n_caption == self.length, "%d != %d" % (
                len(self.video_ids) * n_caption, self.length)

    def __getitem__(self, index):
        cap_id = self.cap_ids[index]
        visual_id = get_visual_id(cap_id)
        #=====================================================================
        # 判断这条数据是视频还是图像
        # 若来自视频
        if visual_id.startswith("video"):
            # video_frames
            # 根据video id 拿到该视频的所有帧的名字
            frame_list = self.video2frames[visual_id]
            # 获取该视频的品牌序号 e.g video4482_0_cls26
            brand_id = int(frame_list[0].split('_')[-1][3:])
            # 根据每一帧的名字取对应的该帧特征(2048-dim）
            frame_vecs = []
            for frame_id in frame_list:
                frame_vecs.append(self.video_feat.read_one(frame_id))
            # 所有帧特征转tensor
            visual_tensor = torch.Tensor(frame_vecs)

        # 若来自图像
        else:
            # 根据数据id拿到图像名
            img_name = self.img_info['idx2img'][int(visual_id[3:])]
            # 根据图像名拿到特征
            vis_feat = []
            # 确定品牌类别
            # insCar数据集写法
            if len(img_name.split('/')) == 2:
                brand_id = int(self.brand_info['cls2idx'][img_name.split('/')[0]])
            # 其他垂直领域
            else:
                brand_id = int(self.brand_info['cls2idx'][img_name.split('/')[-2]])
            img_feat = self.img_feat.read_one(img_name)
            vis_feat.append(img_feat)
            visual_tensor = torch.Tensor(vis_feat)

        # text
        caption = self.captions[cap_id]
        if self.bow2vec is not None:
            # 返回该句描述中出现的单词的词频（one-hot）
            cap_bow = self.bow2vec.mapping(caption)
            if cap_bow is None:
                cap_bow = torch.zeros(self.bow2vec.ndims)
            else:
                cap_bow = torch.Tensor(cap_bow)
        else:
            cap_bow = None

        if self.text_net == 'bi-gru':
            if self.vocab is not None:
                tokens = clean_str(caption)
                caption = []
                caption.append(self.vocab('<start>'))
                caption.extend([self.vocab(token) for token in tokens])
                caption.append(self.vocab('<end>'))
                cap_tensor = torch.Tensor(caption)
            else:
                cap_tensor = None
            return brand_id, visual_tensor, cap_tensor, cap_bow, index, cap_id, visual_id

        elif self.text_net == 'transformers':
            caption = ' '.join(clean_str(caption))
            return brand_id, visual_tensor, caption, cap_bow, index, cap_id, visual_id

    def __len__(self):
        return self.length


class VisDataSet4DualEncoding(data.Dataset):
    """
    Load video_frames frame features by pre-trained CNN model.
    """

    def __init__(self, visual_feat, video2frames=None):
        self.visual_feat = visual_feat
        self.video2frames = video2frames

        self.video_ids = video2frames.keys()
        self.length = len(self.video_ids)

    def __getitem__(self, index):
        video_id = self.video_ids[index]

        frame_list = self.video2frames[video_id]
        frame_vecs = []
        for frame_id in frame_list:
            frame_vecs.append(self.visual_feat.read_one(frame_id))
        frames_tensor = torch.Tensor(frame_vecs)

        return frames_tensor, index, video_id

    def __len__(self):
        return self.length


class TxtDataSet4DualEncoding(data.Dataset):
    """
    Load captions
    """

    def __init__(self, cap_file, bow2vec, vocab):
        # Captions
        self.captions = {}
        self.cap_ids = []
        with open(cap_file, 'r') as cap_reader:
            for line in cap_reader.readlines():
                cap_id, caption = line.strip().split(' ', 1)
                self.captions[cap_id] = caption
                self.cap_ids.append(cap_id)
        self.bow2vec = bow2vec
        self.vocab = vocab
        self.length = len(self.cap_ids)

    def __getitem__(self, index):
        cap_id = self.cap_ids[index]

        caption = self.captions[cap_id]
        if self.bow2vec is not None:
            cap_bow = self.bow2vec.mapping(caption)
            if cap_bow is None:
                cap_bow = torch.zeros(self.bow2vec.ndims)
            else:
                cap_bow = torch.Tensor(cap_bow)
        else:
            cap_bow = None

        if self.vocab is not None:
            tokens = clean_str(caption)
            caption = []
            caption.append(self.vocab('<start>'))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab('<end>'))
            cap_tensor = torch.Tensor(caption)
        else:
            cap_tensor = None

        return cap_tensor, cap_bow, index, cap_id

    def __len__(self):
        return self.length


def get_data_loaders(opt, cap_files, video_feats, img_feats, vocab, bow2vec, text_net, batch_size=100, num_workers=2, n_caption=2,
                     video2frames=None):
    """
    Returns torch.utils.data.DataLoader for train and validation datasets
    Args:
        cap_files: caption files (dict) keys: [train, val]
        visual_feats: image feats (dict) keys: [train, val]
    """
    dset = {'train': Dataset4DualEncoding(opt, cap_files['train'], video_feats['train'], img_feats['train'], bow2vec, vocab,
                                          text_net=text_net, video2frames=video2frames['train']),
            'val': Dataset4DualEncoding(opt, cap_files['val'], video_feats['val'], img_feats['val'], bow2vec, vocab, text_net=text_net,
                                        n_caption=n_caption,
                                        video2frames=video2frames['val']),
            # 新增一个用于检测模型性能的dataloader, 检测模型在训练集上是否过拟合
            'check': Dataset4DualEncoding(opt, cap_files['train'], video_feats['train'], img_feats['train'], bow2vec, vocab, text_net=text_net,
                                          n_caption=n_caption,
                                          video2frames=video2frames['train']),
            'test': Dataset4DualEncoding(opt, cap_files['test'], video_feats['test'], img_feats['test'], bow2vec, vocab, text_net=text_net,
                                         n_caption=n_caption,
                                         video2frames=video2frames['test'])
            }

    collate_fn = None
    if text_net == 'bi-gru':
        collate_fn = collate_frame_gru_fn
    elif text_net == 'transformers':
        collate_fn = collate_frame_transformers_fn

    data_loaders = {x: torch.utils.data.DataLoader(dataset=dset[x],
                                                   batch_size=batch_size,
                                                   shuffle=(x == 'train'),
                                                   pin_memory=True,
                                                   num_workers=num_workers,
                                                   collate_fn=collate_fn)
                    # for x in cap_files}
                    for x in dset.keys()}
    return data_loaders


def get_test_data_loaders(opt, cap_files, video_feats, img_feats, vocab, bow2vec, text_net, batch_size=100, num_workers=2, n_caption=2,
                          video2frames=None):
    """
    Returns torch.utils.data.DataLoader for test dataset
    Args:
        cap_files: caption files (dict) keys: [test]
        visual_feats: image feats (dict) keys: [test]
    """
    dset = {'test': Dataset4DualEncoding(opt, cap_files['test'], video_feats['test'], img_feats['test'], bow2vec, vocab, text_net=text_net,
                                         n_caption=n_caption,
                                         video2frames=video2frames['test'])}

    collate_fn = None
    if text_net == 'bi-gru':
        collate_fn = collate_frame_gru_fn
    elif text_net == 'transformers':
        collate_fn = collate_frame_transformers_fn

    data_loaders = {x: torch.utils.data.DataLoader(dataset=dset[x],
                                                   batch_size=batch_size,
                                                   shuffle=False,
                                                   pin_memory=True,
                                                   num_workers=num_workers,
                                                   collate_fn=collate_fn)
                    for x in cap_files}
    return data_loaders


def get_vis_data_loader(vis_feat, batch_size=100, num_workers=2, video2frames=None):
    dset = VisDataSet4DualEncoding(vis_feat, video2frames)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=num_workers,
                                              collate_fn=collate_frame)
    return data_loader


def get_txt_data_loader(cap_file, vocab, bow2vec, batch_size=100, num_workers=2):
    dset = TxtDataSet4DualEncoding(cap_file, bow2vec, vocab)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=num_workers,
                                              collate_fn=collate_text)
    return data_loader


if __name__ == '__main__':
    pass
