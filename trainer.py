# -*- coding: utf-8 -*-
from __future__ import print_function
import pickle
import os
import sys
import time
import shutil
import json

import torch
from torch.nn.utils import clip_grad_norm_

import evaluator
import util.data_provider as data
from loss import LabLoss, TripletLoss
from loss_ctrs import CrossCLR_onlyIntraModality, ContrastiveLoss
from preprocess.text2vec import get_text_encoder
from preprocess.vocab import Vocabulary  # necessary!

import logging
import tensorboard_logger as tb_logger

import argparse

from util.constant import ROOT_PATH, device
from util.imgbigfile import ImageBigFile
from util.common import makedirsforfile, checkToSkip
from util.util import read_dict, AverageMeter
from util.generic_utils import Progbar
from evaluator import test_post_ranking
from model import FancyRec, get_we_parameter

INFO = __file__


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootpath', type=str, default=ROOT_PATH,
                        help='path to datasets. (default: %s)' % ROOT_PATH)
    parser.add_argument('trainCollection', type=str, help='train collection')
    parser.add_argument('valCollection', type=str, help='validation collection')
    parser.add_argument('testCollection', type=str, help='test collection')
    parser.add_argument('--n_caption', type=int, default=1,
                        help='number of captions of each image/video_frames (default: 1)')
    parser.add_argument('--overwrite', type=int, default=0, choices=[0, 1], help='overwrite existed file. (default: 0)')

    # model
    parser.add_argument('--model', type=str, default='FancyRec', help='model name. (default: FancyRec)')
    parser.add_argument('--measure', type=str, default='cosine', help='measure method. (default: cosine)')
    parser.add_argument('--dropout', default=0.2, type=float, help='dropout rate (default: 0.2)')

    # encoder --- FOR ABLATION STUDY
    parser.add_argument('--concate', type=str, default='full',
                        help='feature concatenation style. (full|reduced) full=level 1+2+3; reduced=level 2+3')
    parser.add_argument('--level_vis', type=str, default='1+2+3', help='ablation study of visual enc')
    parser.add_argument('--level_txt', type=str, default='1+2+3', help='ablation study of text enc')

    # brand
    parser.add_argument('--brand_num', type=int, default=52, help='total brand numbers. default:51')
    parser.add_argument('--brand_aspect', type=int, default=2000, help='brand aspect numbers. default:2000')

    # text-side multi-level encoding
    parser.add_argument('--vocab', type=str, default='word_vocab_5', help='word vocabulary. (default: word_vocab_5)')
    parser.add_argument('--word_dim', type=int, default=500, help='word embedding dimension')
    parser.add_argument('--text_rnn_size', type=int, default=512, help='text rnn encoder size. (default: 1024)')
    parser.add_argument('--text_kernel_num', default=512, type=int, help='number of each kind of text kernel')
    parser.add_argument('--text_kernel_sizes', default='2-3-4', type=str,
                        help='dash-separated kernel size to use for text convolution')
    parser.add_argument('--text_norm', action='store_true', help='normalize the text embeddings at last layer')
    # transformer
    parser.add_argument('--text_transformers_hidden_size', default=768, type=int,
                        help="text transformers encoder size. (default: 768)")
    parser.add_argument('--text_net', type=str, default='transformers',
                        help='(bi-gru|transformers). architecture of text network. default: bi-gru')

    # visual-side multi-level encoding
    parser.add_argument('--video_feature', type=str, default='resnet-152-img1k-flatten0_outputos',
                        help='video feature.')
    parser.add_argument('--img_feature', type=str, default='imgfeat_dim_2048', help='img features.')
    parser.add_argument('--visual_rnn_size', type=int, default=1024, help='visual rnn encoder size')
    parser.add_argument('--visual_kernel_num', default=512, type=int, help='number of each kind of visual kernel')
    parser.add_argument('--visual_kernel_sizes', default='2-3-4-5', type=str,
                        help='dash-separated kernel size to use for visual convolution')
    parser.add_argument('--visual_norm', action='store_true', help='normalize the visual embeddings at last layer')

    # common space learning
    parser.add_argument('--text_mapping_size', type=int, default=512,
                        help='text feature size after processed by text_net. (default:512)')
    parser.add_argument('--visual_mapping_size', type=int, default=2048,
                        help='visual feature size after processed by visual_net. (default:2048)')
    parser.add_argument('--common_embedding_size', type=int, default=2048,
                        help='final embedding size in common space learning.')
    parser.add_argument('--single_modal_visual', action='store_true', help='use visual feature only in fusion stage.')
    parser.add_argument('--single_modal_text', action='store_true', help='use text feature only in fusion stage.')
    parser.add_argument('--fusion_style', type=str, default='fc', help='(fc|mfb). final fusion style between visual. '
                                                                       'and text')
    parser.add_argument('--prj_head_output', action='store_true', help='use the output after projection head'
                                                                       'rather than before the head')
    # loss
    parser.add_argument('--loss_fun', type=str, default='mrl', help='(mrl|CrossCLR) loss function.(default: mrl)')
    parser.add_argument('--margin', type=float, default=0.2, help='rank loss margin')
    parser.add_argument('--direction', type=str, default='all', help='retrieval direction (b2p|p2b|all)')
    parser.add_argument('--max_violation', action='store_true', help='use max instead of sum in the rank loss')
    parser.add_argument('--cost_style', type=str, default='sum', help='cost style (sum, mean). (default: sum)')

    # loss --- FOR ABLATION STUDY
    parser.add_argument('--no_queue', action='store_true', help='do not use queue in Contrastive Loss')
    parser.add_argument('--queue_size', type=int, default=5000, help='queue size in Contrastive Loss')
    parser.add_argument('--no_intra', action='store_true', help='do not use intra-modal loss')

    # optimizer
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer. (default: rmsprop)')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--lr_decay_rate', default=0.99, type=float, help='learning rate decay rate. (default: 0.99)')
    parser.add_argument('--grad_clip', type=float, default=2, help='gradient clipping threshold')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--metric', default='recall', type=str,
                        help='performance metric for validation (mir|recall|auc)')

    # misc
    parser.add_argument('--num_epochs', default=100, type=int, help='Number of training epochs.')
    parser.add_argument('--batch_size', default=128, type=int, help='Size of a training mini-batch.')
    parser.add_argument('--accumulation_step', default=8, type=int, help='Number of gradient accumulation steps.')
    parser.add_argument('--workers', default=0, type=int, help='Number of data loader workers.')
    parser.add_argument('--postfix', default='runs_0', help='Path to save the model and Tensorboard log.')
    parser.add_argument('--log_step', default=10, type=int, help='Number of steps to print and record the log.')
    parser.add_argument('--cv_name', default='FancyRec', type=str, help='')

    args = parser.parse_args()
    return args


def main():
    opt = parse_args()
    print(json.dumps(vars(opt), indent=2))

    rootpath = opt.rootpath
    trainCollection = opt.trainCollection
    valCollection = opt.valCollection
    testCollection = opt.testCollection

    opt.logger_name = os.path.join(rootpath, "model", opt.postfix)

    if checkToSkip(os.path.join(opt.logger_name, 'model_best.pth.tar'), opt.overwrite):
        sys.exit(0)
    if checkToSkip(os.path.join(opt.logger_name, 'val_metric.txt'), opt.overwrite):
        sys.exit(0)
    makedirsforfile(os.path.join(opt.logger_name, 'val_metric.txt'))
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)

    # 准备数据集
    opt.text_kernel_sizes = list(map(int, opt.text_kernel_sizes.split('-')))
    opt.visual_kernel_sizes = list(map(int, opt.visual_kernel_sizes.split('-')))
    collections = {'train': trainCollection, 'val': valCollection, 'test': testCollection}

    cap_file = {'train': '%s.caption.txt' % trainCollection,
                'val': '%s.caption.txt' % valCollection,
                'test': '%s.caption.txt' % testCollection}
    caption_files = {x: os.path.join(rootpath, collections[x], 'TextData', cap_file[x])
                     for x in collections}

    video_feat_path = {x: os.path.join(rootpath, collections[x], 'FeatureData', opt.video_feature)
                       for x in collections}
    img_feat_path = {x: os.path.join(rootpath, collections[x], 'FeatureData', opt.img_feature)
                     for x in collections}

    video_feats = {x: ImageBigFile(video_feat_path[x]) for x in video_feat_path}
    img_feats = {x: ImageBigFile(img_feat_path[x]) for x in img_feat_path}
    opt.visual_feat_dim = video_feats['train'].ndims

    bow_vocab_file = os.path.join(rootpath, opt.trainCollection, 'TextData', 'vocabulary', 'bow', opt.vocab + '.pkl')
    bow_vocab = pickle.load(open(bow_vocab_file, 'rb'))
    bow2vec = get_text_encoder('bow')(bow_vocab)
    opt.bow_vocab_size = len(bow_vocab)

    rnn_vocab_file = os.path.join(rootpath, opt.trainCollection, 'TextData', 'vocabulary', 'rnn', opt.vocab + '.pkl')
    rnn_vocab = pickle.load(open(rnn_vocab_file, 'rb'))
    opt.vocab_size = len(rnn_vocab)

    opt.text_mapping_size = [0, opt.text_mapping_size]
    opt.visual_mapping_size = [0, opt.visual_mapping_size]

    if opt.concate == 'full':
        if opt.text_net == 'bi-gru':
            opt.text_mapping_size[0] = opt.bow_vocab_size + opt.text_rnn_size * 2 + \
                                       opt.text_kernel_num * len(opt.text_kernel_sizes)
        elif opt.text_net == 'transformers':
            opt.text_mapping_size[0] = opt.bow_vocab_size + opt.text_transformers_hidden_size + \
                                       opt.text_kernel_num * len(opt.text_kernel_sizes)

        opt.visual_mapping_size[0] = opt.visual_feat_dim * 2 + opt.visual_rnn_size * 2 + \
                                     opt.visual_kernel_num * len(opt.visual_kernel_sizes)

    elif opt.concate == 'reduced':
        if opt.text_net == 'bi-gru':
            opt.text_mapping_size[0] = 1024

        elif opt.text_net == 'transformers':
            if opt.level_txt == '1+2':
                opt.text_mapping_size[0] = opt.bow_vocab_size + opt.text_transformers_hidden_size
            elif opt.level_txt == '1+3':
                opt.text_mapping_size[0] = opt.bow_vocab_size + opt.text_kernel_num * len(opt.text_kernel_sizes)
            elif opt.level_txt == '2+3':
                opt.text_mapping_size[0] = opt.text_transformers_hidden_size + \
                                           opt.text_kernel_num * len(opt.text_kernel_sizes)
            elif opt.level_txt == '1':
                opt.text_mapping_size[0] = opt.bow_vocab_size
            elif opt.level_txt == '2':
                opt.text_mapping_size[0] = opt.text_transformers_hidden_size
            elif opt.level_txt == '3':
                opt.text_mapping_size[0] = opt.text_kernel_num * len(opt.text_kernel_sizes)
            else:
                opt.text_mapping_size[0] = opt.bow_vocab_size + opt.text_transformers_hidden_size + \
                                           opt.text_kernel_num * len(opt.text_kernel_sizes)

        if opt.level_vis == '1+2':
            opt.visual_mapping_size[0] = opt.visual_feat_dim * 2 + opt.visual_rnn_size * 2
        elif opt.level_vis == '1+3':
            opt.visual_mapping_size[0] = opt.visual_feat_dim * 2 + opt.visual_kernel_num * len(opt.visual_kernel_sizes)
        elif opt.level_vis == '2+3':
            opt.visual_mapping_size[0] = opt.visual_rnn_size * 2 + opt.visual_kernel_num * len(opt.visual_kernel_sizes)
        elif opt.level_vis == '1':
            opt.visual_mapping_size[0] = opt.visual_feat_dim * 2
        elif opt.level_vis == '2':
            opt.visual_mapping_size[0] = opt.visual_rnn_size * 2
        elif opt.level_vis == '3':
            opt.visual_mapping_size[0] = opt.visual_kernel_num * len(opt.visual_kernel_sizes)
        else:
            opt.visual_mapping_size[0] = opt.visual_feat_dim * 2 + opt.visual_rnn_size * 2 + \
                                         opt.visual_kernel_num * len(opt.visual_kernel_sizes)
    else:
        raise NotImplementedError('Unknown concate method: %s' % opt.concate)

    video2frames = {
        x: read_dict(os.path.join(rootpath, collections[x], 'FeatureData', opt.video_feature, 'video2frames.txt'))
        for x in collections}

    # 加载数据
    data_loaders = data.get_data_loaders(opt, caption_files, video_feats, img_feats, rnn_vocab, bow2vec, opt.text_net,
                                         opt.batch_size, opt.workers, opt.n_caption, video2frames=video2frames)

    model = FancyRec(opt).to(device)

    # 参数量检查
    # for name, param in model.named_parameters():
    #     print(name, param.numel())
    #
    # total_num = sum(p.numel() for p in model.parameters())
    # trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print({'Total': total_num, 'Trainable': trainable_num})
    # 检查时后面的代码需注释

    opt.we_parameter = None

    best_rsum = 0
    no_impr_counter = 0
    lr_counter = 0
    best_epoch = None

    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            best_sum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
            model.Eiters = checkpoint['Eiters']
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(opt.resume, start_epoch, best_sum))
            validate(opt, data_loaders['val'], model)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    if opt.optimizer == 'adam':
        opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
    elif opt.optimizer == 'rmsprop':
        opt.optimizer = torch.optim.RMSprop(model.params1, lr=opt.learning_rate)

    # 训练
    for epoch in range(opt.num_epochs):
        train(opt, data_loaders['train'], model, epoch, accumulation_step=opt.accumulation_step)

        print('==========================================================')
        print("=======================Test Phase============================")
        print("==========================================================")
        sum, AUC, NDCG_10, NDCG_50, medR, meanR, r1, r5, r10 = validate(opt, data_loaders['test'], model)

        is_best = sum > best_rsum
        print(' * Current perf in Test: {}'.format(sum))
        print(' * Best perf in Test: {}'.format(best_rsum))

        best_rsum = save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best_rsum': sum,
            'opt': opt,
            'Eiters': model.Eiters,
        }, sum, best_rsum, filename='checkpoint_epoch_%s.pth.tar' % epoch, prefix=opt.logger_name + '/',
            best_epoch=best_epoch)
        if is_best:
            best_epoch = epoch

        lr_counter += 1
        decay_learning_rate(opt.optimizer, opt.lr_decay_rate)

        # 早停
        if not is_best:
            no_impr_counter += 1
            if no_impr_counter > 10:
                print('Early stopping happened.\n')
                break
            # 学习率调整
            if lr_counter > 2:
                decay_learning_rate(opt.optimizer, 0.5)
                lr_counter = 0
        else:
            no_impr_counter = 0

    print('best performance on Val: {}\n'.format(best_rsum))


def train(opt, train_loader, model, epoch, accumulation_step=8, lr_scheduler=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.brand_encoding.train()
    if opt.single_modal_text:
        model.text_encoding.train()
    elif opt.single_modal_visual:
        model.vid_encoding.train()
    else:
        model.vid_encoding.train()
        model.text_encoding.train()
        model.fusion_encoding.train()

    progbar = Progbar(len(train_loader.dataset))
    end = time.time()
    train_loss = []

    loss_func = None
    if opt.loss_fun == 'CrossCLR':
        loss_func = CrossCLR_onlyIntraModality().to(device)
    elif opt.loss_fun == 'mrl':
        loss_func = TripletLoss(margin=opt.margin,
                                max_violation=opt.max_violation,
                                cost_style=opt.cost_style,
                                direction=opt.direction,
                                loss_fun=opt.loss_fun).to(device)
    elif opt.loss_fun == 'cl':
        loss_func = ContrastiveLoss(opt=opt).to(device)
    elif opt.loss_fun == 'lab':
        loss_func = LabLoss()

    print('Epoch[{0} / {1}] LR: {2}'.format(epoch, opt.num_epochs, get_learning_rate(opt.optimizer)[0]))

    for i, train_data in enumerate(train_loader):
        data_time.update(time.time() - end)

        brand_ids = train_data[0]
        videos = train_data[1]
        captions = train_data[2]

        model.Eiters += 1

        brand_emb, post_emb = model(brand_ids, videos, captions)

        loss = None
        if opt.loss_fun == 'CrossCLR':
            loss = loss_func(brand_emb, post_emb)
        elif opt.loss_fun == 'mrl':
            loss = loss_func(brand_ids, brand_emb, post_emb)
        elif opt.loss_fun == 'cl':
            loss = loss_func(brand_emb, post_emb)
        elif opt.loss_fun == 'lab':
            loss = loss_func(brand_emb)

        train_loss.append(loss.item())

        loss.backward()
        if (i + 1) % accumulation_step == 0:
            if opt.grad_clip > 0:
                clip_grad_norm_(model.parameters(), opt.grad_clip)
            opt.optimizer.step()
            opt.optimizer.zero_grad()

        progbar.add(post_emb.size(0), values=[('loss', loss.item())])
        batch_time.update(time.time() - end)
        end = time.time()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return train_loss


def validate(opt, val_loader, model):

    brands, post_embs = evaluator.encode_data(model, val_loader, opt.log_step, logging.info)
    MedR, MeanR, AUC, NDCG_10, NDCG_50, r1, r5, r10 = test_post_ranking(opt.brand_num, opt.metric, model, post_embs,
                                                                        brands)
    print('MedR:', MedR)
    print('MeanR:', MeanR)
    print('AUC[0-1]:', AUC)
    print('NDCG@10[0-1]:', NDCG_10)
    print('NDCG@50[0-1]:', NDCG_50)
    print('recall@1:', r1)
    print('recall@5:', r5)
    print('recall@10:', r10)

    sum = 0.0
    sum += ((AUC + NDCG_10 + NDCG_50) * 100 + r1 + r5 + r10)
    return sum, AUC, NDCG_10, NDCG_50, MedR, MeanR, r1, r5, r10


def save_checkpoint(state, sum, best_rsum, filename='checkpoint.pth.tar', prefix='', best_epoch=None):
    if best_epoch is None or sum > best_rsum * 0.99:
        torch.save(state, prefix + filename)
    if sum > best_rsum:
        shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')
    return max(sum, best_rsum)


def decay_learning_rate(optimizer, decay):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay


def get_learning_rate(optimizer):
    lr_list = []
    for param_group in optimizer.param_groups:
        lr_list.append(param_group['lr'])
    return lr_list


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
