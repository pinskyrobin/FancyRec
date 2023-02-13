# encoding:utf-8
from __future__ import print_function
import pickle
import os

import torch

import evaluator
from util.imgbigfile import ImageBigFile
from FGMCD import FGMCD
import util.data_provider as data
from preprocess.text2vec import get_text_encoder

import logging
import json

import argparse
from util.util import read_dict
from util.constant import ROOT_PATH
from util.common import makedirsforfile, checkToSkip
from evaluator import test_post_ranking
import sys

"""单独跑测试
"""
def parse_args():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('testCollection', type=str, help='test collection')
    parser.add_argument('--rootpath', type=str, default=ROOT_PATH, help='path to datasets. (default: %s)'%ROOT_PATH)
    parser.add_argument('--overwrite', type=int, default=0, choices=[0,1],  help='overwrite existed file. (default: 0)')
    parser.add_argument('--log_step', default=10, type=int, help='Number of steps to print and record the log.')
    parser.add_argument('--batch_size', default=128, type=int, help='Size of a training mini-batch.')
    parser.add_argument('--workers', default=0, type=int, help='Number of data loader workers.')
    parser.add_argument('--logger_name', default='runs', help='Path to save the model and Tensorboard log.')
    parser.add_argument('--checkpoint_name', default='model_best.pth.tar', type=str, help='name of checkpoint (default: model_best.pth.tar)')
    parser.add_argument('--n_caption', type=int, default=20, help='number of captions of each image/video_frames (default: 1)')
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
    # print(resume)
    if not os.path.exists(resume):
        logging.info(resume + ' not exists.')
        sys.exit(0)

    # from functools import partial
    # import pickle
    # pickle.load = partial(pickle.load, encoding="latin1")
    # pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
    # checkpoint = torch.load(resume, map_location=lambda storage, loc: storage, pickle_module=pickle)
    checkpoint = torch.load(resume)
    start_epoch = checkpoint['epoch']
    best_rsum = checkpoint['best_rsum']
    print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
          .format(resume, start_epoch, best_rsum))
    options = checkpoint['opt']
    if not hasattr(options, 'concate'):
        setattr(options, "concate", "full")

    trainCollection = options.trainCollection
    output_dir = resume.replace(trainCollection, testCollection)
    output_dir = output_dir.replace('/%s/' % options.cv_name, '/results/%s/' % trainCollection )
    pred_error_matrix_file = os.path.join(output_dir, 'pred_errors_matrix.pth.tar')
    # print(pred_error_matrix_file)
    if checkToSkip(pred_error_matrix_file, opt.overwrite):
        sys.exit(0)
    makedirsforfile(pred_error_matrix_file)

    # data loader prepare
    caption_files = {'test': os.path.join(rootpath, testCollection, 'TextData', '%s.caption.txt' %testCollection)}
    # options变量是从检查点中取出
    video_feat_path = os.path.join(rootpath, testCollection, 'FeatureData', options.video_feature)
    img_feat_path = os.path.join(rootpath, testCollection, 'FeatureData', options.img_feature)
    video_feats = {'test': ImageBigFile(video_feat_path)}
    img_feats = {'test': ImageBigFile(img_feat_path)}
    assert options.visual_feat_dim == video_feats['test'].ndims
    video2frames = {'test': read_dict(os.path.join(rootpath, testCollection, 'FeatureData', options.video_feature, 'video2frames.txt'))}

    # set bow vocabulary and encoding
    bow_vocab_file = os.path.join(rootpath, options.trainCollection, 'TextData', 'vocabulary', 'bow', options.vocab+'.pkl')
    bow_vocab = pickle.load(open(bow_vocab_file, 'rb'))
    bow2vec = get_text_encoder('bow')(bow_vocab)
    # print("bow2vec:", len(bow_vocab))
    # print(bow_vocab)
    options.bow_vocab_size = len(bow_vocab)

    # set rnn vocabulary 
    rnn_vocab_file = os.path.join(rootpath, options.trainCollection, 'TextData', 'vocabulary', 'rnn', options.vocab+'.pkl')
    rnn_vocab = pickle.load(open(rnn_vocab_file, 'rb'))
    options.vocab_size = len(rnn_vocab)
    # print("rnn_vocalb:", len(rnn_vocab))
    print("prepare dataloader..")
    # set data loader
    data_loader = data.get_test_data_loaders(opt,
        caption_files, video_feats, img_feats, rnn_vocab, bow2vec, options.text_net, opt.batch_size, opt.workers, opt.n_caption, video2frames=video2frames)

    # Construct the model
    model = FGMCD(options)
    model.load_state_dict(checkpoint['model'])
    # parallel
    # model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4])
    # model = nn.parallel.data_parallel(model, device_ids=[0, 1, 2, 3, 4])
    model.Eiters = checkpoint['Eiters']

    brands, post_embs = evaluator.encode_data(model, data_loader['test'], opt.log_step, logging.info)

    ranking_metrics = test_post_ranking(options.brand_num, options.metric, model, post_embs, brands)
    print('MedR:', ranking_metrics[0])
    print('MeanR:', ranking_metrics[1])
    print('AUC[0-1]:', ranking_metrics[2])
    print('NDCG@10[0-1]:', ranking_metrics[3])
    print('NDCG@50[0-1]:', ranking_metrics[4])
    print('recall@1:', ranking_metrics[5])
    print('recall@5:', ranking_metrics[6])
    print('recall@10:', ranking_metrics[7])


if __name__ == '__main__':
    main()
