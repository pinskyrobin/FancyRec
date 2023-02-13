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
from loss_exp import CrossCLR_onlyIntraModality
from preprocess.text2vec import get_text_encoder
from preprocess.vocab import Vocabulary

import logging
import tensorboard_logger as tb_logger

import argparse

from util.constant import ROOT_PATH, device
from util.imgbigfile import ImageBigFile
from util.common import makedirsforfile, checkToSkip
from util.util import read_dict, AverageMeter, LogCollector
from util.generic_utils import Progbar
from evaluator import test_post_ranking
from FGMCD import FGMCD, get_we_parameter

INFO = __file__

"""训练用代码文件
"""


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
    parser.add_argument('--model', type=str, default='dual_encoding', help='model name. (default: dual_encoding)')
    parser.add_argument('--concate', type=str, default='full',
                        help='feature concatenation style. (full|reduced) full=level 1+2+3; reduced=level 2+3')
    parser.add_argument('--measure', type=str, default='cosine', help='measure method. (default: cosine)')
    parser.add_argument('--dropout', default=0.2, type=float, help='dropout rate (default: 0.2)')
    # brand
    parser.add_argument('--brand_num', type=int, default=52, help='total brand numbers. default:51')
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
    # loss
    parser.add_argument('--loss_fun', type=str, default='mrl', help='(mrl|eet) loss function.(default: mrl)')
    parser.add_argument('--margin', type=float, default=0.2, help='rank loss margin')
    parser.add_argument('--direction', type=str, default='all', help='retrieval direction (b2p|p2b|all)')
    parser.add_argument('--max_violation', action='store_true', help='use max instead of sum in the rank loss')
    parser.add_argument('--cost_style', type=str, default='sum', help='cost style (sum, mean). (default: sum)')
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
    parser.add_argument('--workers', default=0, type=int, help='Number of data loader workers.')
    parser.add_argument('--postfix', default='runs_0', help='Path to save the model and Tensorboard log.')
    parser.add_argument('--log_step', default=10, type=int, help='Number of steps to print and record the log.')
    parser.add_argument('--cv_name', default='FGMCD', type=str, help='')

    args = parser.parse_args()
    return args


def main():
    opt = parse_args()
    print(json.dumps(vars(opt), indent=2))

    rootpath = opt.rootpath
    trainCollection = opt.trainCollection
    valCollection = opt.valCollection
    testCollection = opt.testCollection

    # margin ranking loss and cosine similarity
    if opt.loss_fun == "mrl" and opt.measure == "cosine":
        assert opt.text_norm is True
        assert opt.visual_norm is True

    # mark reduce method if concate=reduced
    reduced = ""
    if opt.concate == "reduced":
        reduced = "2_3"
    print("max_violation", opt.max_violation)

    # checkpoint path
    model_info = '%s_concate_%s_dp_%.1f_measure_%s' % (opt.model, opt.concate + reduced, opt.dropout, opt.measure)
    # text-side multi-level encoding info
    text_encode_info = 'vocab_%s_word_dim_%s_text_rnn_size_%s_text_norm_%s_text_net_%s' % \
                       (opt.vocab, opt.word_dim, opt.text_rnn_size, opt.text_norm, opt.text_net)
    text_encode_info += "_kernel_sizes_%s_num_%s" % (opt.text_kernel_sizes, opt.text_kernel_num)  # cnn有234
    # video_frames-side multi-level encoding info
    visual_encode_info = 'video_feature_%s_img_feature_%s_visual_rnn_size_%d_visual_norm_%s' % \
                         (opt.video_feature, opt.img_feature, opt.visual_rnn_size, opt.visual_norm)
    visual_encode_info += "_kernel_sizes_%s_num_%s" % (opt.visual_kernel_sizes, opt.visual_kernel_num)  # cnn有2345
    # common space learning info
    mapping_info = "mapping_text_%s_img_%s" % (opt.text_mapping_size, opt.visual_mapping_size)
    loss_info = 'loss_func_%s_margin_%s_direction_%s_max_violation_%s_cost_style_%s' % \
                (opt.loss_fun, opt.margin, opt.direction, opt.max_violation, opt.cost_style)
    optimizer_info = 'optimizer_%s_lr_%s_decay_%.2f_grad_clip_%.1f_val_metric_%s' % \
                     (opt.optimizer, opt.learning_rate, opt.lr_decay_rate, opt.grad_clip, opt.metric)

    # ==================================================
    # 根据使用模态的不同 相应地修改log的保存位置
    if opt.single_modal_visual:
        modalities = "single_modal_visual"
    elif opt.single_modal_text:
        modalities = "single_modal_text"
    else:
        modalities = "visual_plus_text"
    opt.logger_name = os.path.join(rootpath, trainCollection, opt.cv_name, valCollection, model_info, text_encode_info,
                                   visual_encode_info, mapping_info, loss_info, optimizer_info, opt.postfix, modalities)

    # exit if file exists and overwrite=0
    if checkToSkip(os.path.join(opt.logger_name, 'model_best.pth.tar'), opt.overwrite):
        sys.exit(0)
    if checkToSkip(os.path.join(opt.logger_name, 'val_metric.txt'), opt.overwrite):
        sys.exit(0)
    makedirsforfile(os.path.join(opt.logger_name, 'val_metric.txt'))
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)

    # get kernel size
    # text-kernel [2,3,4]
    opt.text_kernel_sizes = list(map(int, opt.text_kernel_sizes.split('-')))
    # visual-kernel [2,3,4,5]
    opt.visual_kernel_sizes = list(map(int, opt.visual_kernel_sizes.split('-')))
    # collections: train, val
    collections = {'train': trainCollection, 'val': valCollection, 'test': testCollection}
    cap_file = {'train': '%s.caption.txt' % trainCollection,
                'val': '%s.caption.txt' % valCollection,
                'test': '%s.caption.txt' % testCollection}
    # get train&val caption
    caption_files = {x: os.path.join(rootpath, collections[x], 'TextData', cap_file[x])
                     for x in collections}
    # Load visual features
    # opt.visual_feature : 'resnet-152-img1k-flatten0_outputos'
    video_feat_path = {x: os.path.join(rootpath, collections[x], 'FeatureData', opt.video_feature)
                       for x in collections}
    img_feat_path = {x: os.path.join(rootpath, collections[x], 'FeatureData', opt.img_feature)
                     for x in collections}
    # wrap video features
    video_feats = {x: ImageBigFile(video_feat_path[x]) for x in video_feat_path}
    # wrap image features
    img_feats = {x: ImageBigFile(img_feat_path[x]) for x in img_feat_path}
    # get video feature dimension(2048)
    opt.visual_feat_dim = video_feats['train'].ndims

    # set bow vocabulary and encoding
    bow_vocab_file = os.path.join(rootpath, opt.trainCollection, 'TextData', 'vocabulary', 'bow', opt.vocab + '.pkl')
    bow_vocab = pickle.load(open(bow_vocab_file, 'rb'))
    # wrap bow encoding
    bow2vec = get_text_encoder('bow')(bow_vocab)
    opt.bow_vocab_size = len(bow_vocab)
    print("bow_vocab_size", len(bow_vocab))  # 4318

    # set rnn vocabulary
    rnn_vocab_file = os.path.join(rootpath, opt.trainCollection, 'TextData', 'vocabulary', 'rnn', opt.vocab + '.pkl')
    # Vocabulary object
    rnn_vocab = pickle.load(open(rnn_vocab_file, 'rb'))
    opt.vocab_size = len(rnn_vocab)
    # print(len(rnn_vocab)) 7811

    # initialize word embedding
    opt.we_parameter = None
    if opt.word_dim == 500:
        # 用flickr的词库来做词向量
        w2v_data_path = os.path.join(rootpath, "word2vec", 'flickr', 'vec500flickr30m')
        # 将自己词袋中的词做成词向量
        opt.we_parameter = get_we_parameter(rnn_vocab, w2v_data_path)  # 获得每个词的向量表示

    # mapping layer structure [0,2048]
    opt.text_mapping_size = [0, opt.text_mapping_size]
    opt.visual_mapping_size = [0, opt.visual_mapping_size]

    if opt.concate == 'full':  # 多级特征的拼接方式
        # 最终concat得到的文本特征dim
        if opt.text_net == 'bi-gru':
            opt.text_mapping_size[0] = opt.bow_vocab_size + opt.text_rnn_size * 2 + \
                                       opt.text_kernel_num * len(
                opt.text_kernel_sizes)
            # print(opt.text_mapping_layers[0])  # 6878
        elif opt.text_net == 'transformers':
            opt.text_mapping_size[0] = opt.bow_vocab_size + opt.text_transformers_hidden_size + \
                                       opt.text_kernel_num * len(opt.text_kernel_sizes)

        # 最终concat得到的视觉特征dim
        opt.visual_mapping_size[0] = opt.visual_feat_dim * 2 + opt.visual_rnn_size * 2 + \
                                     opt.visual_kernel_num * len(opt.visual_kernel_sizes)

    # 如果是reduced的话 这里设置投影矩阵的维度要复杂一些
    # 因为跟具体的reduced方式有关, reduced可以是以下这几种：
    # level 1
    # level 2
    # level 3
    # level 1+2
    # level 1+3
    # level 2+3
    # concate方式是full的话代表 level 1 + 2 + 3
    elif opt.concate == 'reduced':
        if opt.text_net == 'bi-gru':
            # opt.text_mapping_size[0] = opt.text_rnn_size * 2 + opt.text_kernel_num * len(opt.text_kernel_sizes)
            # level 1+3
            # opt.text_mapping_size[0] = 5854
            # level 2+3
            opt.text_mapping_size[0] = 2560

        elif opt.text_net == 'transformers':
            opt.text_mapping_size[0] = opt.text_transformers_hidden_size + opt.text_kernel_num * len(
                opt.text_kernel_sizes)
            # opt.text_mapping_size[0] = opt.bow_vocab_size + opt.text_transformers_hidden_size + \
            #                            opt.text_kernel_num * len(opt.text_kernel_sizes)
            # print("bow_vocab_size",opt.bow_vocab_size)
            # print("text_transformers_hidden_size",opt.text_transformers_hidden_size)
            # print("text_kernel_num * len(opt.text_kernel_sizes",opt.text_kernel_num * len(opt.text_kernel_sizes))

        opt.visual_mapping_size[0] = opt.visual_feat_dim + opt.visual_rnn_size * 2 + opt.visual_kernel_num * len(
            opt.visual_kernel_sizes)
        # print(opt.text_mapping_layers[0])  # 2560
        # print(opt.visual_mapping_layers[0])  # 6144
    else:
        raise NotImplementedError('Model %s not implemented' % opt.model)

    # set data loader # 得到数据
    video2frames = {
        x: read_dict(os.path.join(rootpath, collections[x], 'FeatureData', opt.video_feature, 'video2frames.txt'))
        for x in collections}
    # ==================================dataloader============================
    # wmy
    data_loaders = data.get_data_loaders(opt, caption_files, video_feats, img_feats, rnn_vocab, bow2vec, opt.text_net,
                                         opt.batch_size, opt.workers, opt.n_caption,
                                         video2frames=video2frames)

    # Construct the model
    model = FGMCD(opt).to(device)

    # parallel
    # model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    # model = model.module
    opt.we_parameter = None

    # Train the Model
    best_rsum = 0
    no_impr_counter = 0
    lr_counter = 0
    best_epoch = None

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            best_sum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
            # Eiters is used to show logs as the continuation of another
            # training
            model.Eiters = checkpoint['Eiters']
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(opt.resume, start_epoch, best_sum))
            validate(opt, data_loaders['val'], model)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    fout_val_metric_hist = open(os.path.join(opt.logger_name, 'val_metric_hist.txt'), 'w')
    total_loss = []
    val_auc = []
    test_auc = []
    for epoch in range(opt.num_epochs):
        # train for one epoch
        train_loss = train(opt, data_loaders['train'], model, epoch)
        total_loss.extend(train_loss)

        # evaluate on validation set
        # print('==========================================================')
        # print("===============Validation Phase===========================")
        # print("==========================================================")
        # sum, AUC, NDCG_10, NDCG_50, medR, meanR, r1, r5, r10 = \
        # validate(opt, data_loaders['val'], model, measure=opt.measure)
        # val_auc.append((AUC, NDCG_10, NDCG_50))

        # remember best R@ sum and save checkpoint
        # is_best = sum > best_rsum
        # best_rsum = max(sum, best_rsum)
        # print(' * Current perf in Val: {}'.format(sum))
        # print(' * Best perf in Val: {}'.format(best_rsum))
        # fout_val_metric_hist.write('epoch_%d: %f\n' % (epoch, sum))
        # fout_val_metric_hist.flush()

        # print('==========================================================')
        # print("===============检查在训练集上拟合的情况===================")
        # print("==========================================================")
        # validate(opt, data_loaders['check'], model, measure=opt.measure)

        print('==========================================================')
        print("=======================Test Phase============================")
        print("==========================================================")
        sum, AUC, NDCG_10, NDCG_50, medR, meanR, r1, r5, r10 = validate(opt, data_loaders['test'], model)
        test_res_str = "sum: " + str(sum) + "\nAUC: " + str(AUC) + "\nNDCG_10: " + str(NDCG_10) + "\nNDCG_50: " + str(
            NDCG_50) + "\nMedR: " + str(medR) + \
                       "\nMeanR: " + str(meanR) + "\nr1: " + str(r1) + "\nr5: " + str(r5) + "\nr10: " + str(
            r10) + "\n\n\n\n\n"
        test_auc.append(test_res_str)

        is_best = sum > best_rsum
        best_rsum = max(sum, best_rsum)
        print(' * Current perf in Test: {}'.format(sum))
        print(' * Best perf in Test: {}'.format(best_rsum))

        if is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'best_rsum': best_rsum,
                'opt': opt,
                'Eiters': model.Eiters,
            }, is_best, filename='checkpoint_epoch_%s.pth.tar' % epoch, prefix=opt.logger_name + '/',
                best_epoch=best_epoch)
            best_epoch = epoch

        lr_counter += 1
        decay_learning_rate(opt, opt.optimizer, opt.lr_decay_rate)
        # Early Stopping.
        if not is_best:
            # Early stop occurs if the validation performance does not improve in ten consecutive epochs
            no_impr_counter += 1
            if no_impr_counter > 80:
                print('Early stopping happened.\n')
                break

            # When the validation performance decreased after an epoch,
            # we divide the learning rate by 2 and continue training;
            # but we use each learning rate for at least 3 epochs.
            if lr_counter > 2:
                decay_learning_rate(opt, opt.optimizer, 0.5)
                lr_counter = 0
        else:
            no_impr_counter = 0

    fout_val_metric_hist.close()
    """save loss and auc in file"""
    with open('out/loss.txt', 'w') as f:
        f.write(str(total_loss))
    with open('out/val_auc.txt', 'w') as f:
        f.write(str(val_auc))
    with open('out/test_total_res.txt', 'w') as f:
        f.write(str(test_auc))

    print('best performance on Val: {}\n'.format(best_rsum))
    with open(os.path.join(opt.logger_name, 'val_metric.txt'), 'w') as fout:
        fout.write('best performance on validation: ' + str(best_rsum))

    # generate evaluation shell script
    if testCollection == 'iacc.3':
        templete = ''.join(open('bin/TEMPLATE_do_predict.sh').readlines())
        script_str = templete.replace('@@@query_sets@@@', 'tv16.avs.txt,tv17.avs.txt,tv18.avs.txt')
    else:
        templete = ''.join(open('bin/TEMPLATE_do_test.sh', 'r', encoding='utf8').readlines())
        script_str = templete.replace('@@@n_caption@@@', str(opt.n_caption))
    script_str = script_str.replace('@@@rootpath@@@', rootpath)
    script_str = script_str.replace('@@@testCollection@@@', testCollection)
    script_str = script_str.replace('@@@logger_name@@@', opt.logger_name)
    script_str = script_str.replace('@@@overwrite@@@', str(opt.overwrite))

    # perform evaluation on test set
    runfile = 'do_test_%s_%s.sh' % (opt.model, testCollection)
    open(runfile, 'w', encoding='utf8').write(script_str + '\n')
    os.system('chmod +x %s' % runfile)
    # os.system('./'+runfile)


def train(opt, train_loader, model, epoch):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    # switch to train mode
    # TODO
    # model.vid_encoding.train()
    # model.text_encoding.train()
    # model.brand_encoding.train()
    # model.fusion_encoding.train()
    model.train()

    # optimize brand-net first
    progbar = Progbar(len(train_loader.dataset))
    end = time.time()
    train_loss = []

    # loss_func = TripletLoss(margin=opt.margin,
    #                         max_violation=opt.max_violation,
    #                         cost_style=opt.cost_style,
    #                         direction=opt.direction,
    #                         loss_fun=opt.loss_fun).to(device)

    loss_func = CrossCLR_onlyIntraModality(logger=train_logger).to(device)

    # loss_func = LabLoss().to(device)

    optimizer = None
    if opt.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
    elif opt.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.params1, lr=opt.learning_rate)

    print('Epoch[{0} / {1}] LR: {2}'.format(epoch, opt.num_epochs, get_learning_rate(optimizer)[0]))

    for i, train_data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        brand_ids = train_data[0].to(device)
        videos = train_data[1]
        captions = train_data[2]

        model.logger = train_logger
        model.Eiters += 1
        model.logger.update('Eit', model.Eiters)
        model.logger.update('lr', optimizer.param_groups[0]['lr'])

        # Update the model
        videos, videos_origin, vis_lengths, vidoes_mask = videos
        cap_wids, cap_bows, txt_lengths, cap_mask = captions
        if cap_wids is not None:
            cap_wids = cap_wids.to(device)
        if cap_bows is not None:
            cap_bows = cap_bows.to(device)
        if cap_mask is not None:
            cap_mask = cap_mask.to(device)

        brand_ids = brand_ids.to(device)
        videos = videos.to(device)
        videos_origin = videos_origin.to(device)
        vidoes_mask = vidoes_mask.to(device)

        brand_emb, post_emb = model(brand_ids,
                                    videos, videos_origin, vis_lengths, vidoes_mask,
                                    cap_wids, cap_bows, txt_lengths, cap_mask)

        optimizer.zero_grad()
        loss = loss_func(brand_emb, post_emb)
        # loss = loss_func(brand_ids, brand_emb, post_emb)
        # loss = loss_func(brand_emb)
        train_loss.append(loss.item())
        model.logger.update('Le', loss.item(), brand_emb.size(0))

        loss.backward()
        if opt.grad_clip > 0:
            clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()

        progbar.add(post_emb.size(0), values=[('loss', loss.item())])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Record logs in tensorboard
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)
    return train_loss


def validate(opt, val_loader, model):
    # compute the encoding for all the validation video_frames and captions
    # return brands' index and post embeddings in validation set
    brands, post_embs = evaluator.encode_data(model, val_loader, opt.log_step, logging.info)

    # we load data as video_frames-sentence pairs,
    # but we only need to forward each video_frames once for evaluation,
    # so we get the video_frames set and mask out same videos with feature_mask
    # feature_mask = []
    # evaluate_videos = set()
    # for video_id in video_ids:
    #     feature_mask.append(video_id not in evaluate_videos)
    #     evaluate_videos.add(video_id)
    # video_embs = video_embs[feature_mask]
    # video_ids = [x for idx, x in enumerate(video_ids) if feature_mask[idx] is True]
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

    # record metrics in tensorboard
    tb_logger.log_value('MedR', MedR, step=model.Eiters)
    tb_logger.log_value('MeanR', MeanR, step=model.Eiters)
    tb_logger.log_value('AUC', AUC, step=model.Eiters)
    tb_logger.log_value('NDCG@10', NDCG_10, step=model.Eiters)
    tb_logger.log_value('NDCG@50', NDCG_50, step=model.Eiters)
    tb_logger.log_value('r1', r1, step=model.Eiters)
    tb_logger.log_value('r5', r5, step=model.Eiters)
    tb_logger.log_value('r10', r10, step=model.Eiters)

    sum = 0.0
    sum += (r1 + r5 + r10)
    return sum, AUC, NDCG_10, NDCG_50, MedR, MeanR, r1, r5, r10


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix='', best_epoch=None):
    """save checkpoint at specific path"""
    torch.save(state, prefix + filename)
    if is_best:
        shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')
    if best_epoch is not None:
        os.remove(prefix + 'checkpoint_epoch_%s.pth.tar' % best_epoch)


def decay_learning_rate(opt, optimizer, decay):
    """decay learning rate to the last LR"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay


def get_learning_rate(optimizer):
    """Return learning rate"""
    lr_list = []
    for param_group in optimizer.param_groups:
        lr_list.append(param_group['lr'])
    return lr_list


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
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
