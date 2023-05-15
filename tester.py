# -*- coding: utf-8 -*-
from __future__ import print_function
import pickle
import os

import torch

import evaluator
from util.imgbigfile import ImageBigFile
from model import FancyRec
import util.data_provider as data
from preprocess.text2vec import get_text_encoder
from preprocess.vocab import Vocabulary

import logging
import json

import argparse
from util.util import read_dict
from util.constant import ROOT_PATH, device
from util.common import makedirsforfile, checkToSkip
from evaluator import test_post_ranking
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('testCollection', type=str, help='test collection')
    parser.add_argument('--rootpath', type=str, default=ROOT_PATH, help='path to datasets. (default: %s)' % ROOT_PATH)
    parser.add_argument('--overwrite', type=int, default=0, choices=[0, 1], help='overwrite existed file. (default: 0)')
    parser.add_argument('--log_step', default=10, type=int, help='Number of steps to print and record the log.')
    parser.add_argument('--batch_size', default=128, type=int, help='Size of a training mini-batch.')
    parser.add_argument('--workers', default=0, type=int, help='Number of data loader workers.')
    parser.add_argument('--logger_name', default='runs', help='Path to save the model and Tensorboard log.')
    parser.add_argument('--checkpoint_name', default='model_best.pth.tar', type=str,
                        help='name of checkpoint (default: model_best.pth.tar)')
    parser.add_argument('--n_caption', type=int, default=20,
                        help='number of captions of each image/video_frames (default: 1)')
    parser.add_argument('--level_vis', type=str, default='1+2+3', help='ablation study of visual enc')
    parser.add_argument('--level_txt', type=str, default='1+2+3', help='ablation study of text enc')
    args = parser.parse_args()
    return args


def load_config(config_path):
    variables = {}
    exec(compile(open(config_path, "rb").read(), config_path, 'exec'), variables)
    return variables['config']


def main():
    opt = parse_args()
    print(json.dumps(vars(opt), indent=2))

    rootpath = opt.rootpath
    testCollection = opt.testCollection
    n_caption = opt.n_caption
    resume = os.path.join(opt.logger_name, opt.checkpoint_name)
    if not os.path.exists(resume):
        logging.info(resume + ' not exists.')
        sys.exit(0)

    checkpoint = torch.load(resume)
    print("=> loaded!")
    options = checkpoint['opt']
    if not hasattr(options, 'concate'):
        setattr(options, "concate", "full")

    trainCollection = options.trainCollection
    output_dir = resume.replace(trainCollection, testCollection)
    output_dir = output_dir.replace('/%s/' % options.cv_name, '/results/%s/' % trainCollection)
    pred_error_matrix_file = os.path.join(output_dir, 'pred_errors_matrix.pth.tar')
    if checkToSkip(pred_error_matrix_file, opt.overwrite):
        sys.exit(0)
    makedirsforfile(pred_error_matrix_file)

    caption_files = {'test': os.path.join(rootpath, testCollection, 'TextData', '%s.caption.txt' % testCollection)}
    video_feat_path = os.path.join(rootpath, testCollection, 'FeatureData', options.video_feature)
    img_feat_path = os.path.join(rootpath, testCollection, 'FeatureData', options.img_feature)
    video_feats = {'test': ImageBigFile(video_feat_path)}
    img_feats = {'test': ImageBigFile(img_feat_path)}
    assert options.visual_feat_dim == video_feats['test'].ndims
    video2frames = {'test': read_dict(
        os.path.join(rootpath, testCollection, 'FeatureData', options.video_feature, 'video2frames.txt'))}

    bow_vocab_file = os.path.join(rootpath, options.trainCollection, 'TextData', 'vocabulary', 'bow',
                                  options.vocab + '.pkl')
    bow_vocab = pickle.load(open(bow_vocab_file, 'rb'))
    bow2vec = get_text_encoder('bow')(bow_vocab)
    options.bow_vocab_size = len(bow_vocab)

    rnn_vocab_file = os.path.join(rootpath, options.trainCollection, 'TextData', 'vocabulary', 'rnn',
                                  options.vocab + '.pkl')
    rnn_vocab = pickle.load(open(rnn_vocab_file, 'rb'))
    options.vocab_size = len(rnn_vocab)
    print("=> prepare dataloader..")
    data_loader = data.get_test_data_loaders(opt,
                                             caption_files, video_feats, img_feats, rnn_vocab, bow2vec,
                                             options.text_net, opt.batch_size, opt.workers, opt.n_caption,
                                             video2frames=video2frames)

    model = FancyRec(options).to(device)
    model.load_state_dict(checkpoint['model'])
    model.Eiters = checkpoint['Eiters']

    brands, post_embs = evaluator.encode_data(model, data_loader['test'], opt.log_step, logging.info)

    ranking_metrics = test_post_ranking(options.brand_num, options.metric, model, post_embs, brands)

    print('AUC[0-1]:', ranking_metrics[2])
    print('NDCG@10[0-1]:', ranking_metrics[3])
    print('NDCG@50[0-1]:', ranking_metrics[4])
    print('recall@1:', ranking_metrics[5])


if __name__ == '__main__':
    main()
