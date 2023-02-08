# -*- coding: utf-8 -*
import cv2
import numpy as np
import array
import torch
import torch.nn as nn
import os
import json
from basic.util import read_dict
import evaluation
import io
import codecs
from transformers import BertTokenizer, BertModel, BertConfig
# from torch.utils.tensorboard import SummaryWriter
import random

def test():
    f = open("video_captions.txt", 'r').read()
    caps = json.loads(f)
    print(caps)
    print(len(caps))  # 10889 有描述的视频数
    video_info = read_dict("video_info.txt")
    # print(caps.keys())
    print(len(video_info["video2idx"].keys()))  # 11067
    print(len(video_info['idx2video'].keys()))
    f = open('cls.txt', 'r').read()
    video_cls = json.loads(f)
    print(video_cls)


def read_one_frame_feature(dim):
    index_name_array = [(0, 'haha')]
    index_name_array.sort(key=lambda v: v[0])
    sorted_index = [x[0] for x in index_name_array]
    # print(sorted_index)
    nr_of_images = len(index_name_array)
    offset = np.float32(1).nbytes * dim
    # read binary file
    binary_file = '/root/CV/dual_encoding_wu/resnet152_img_2048/feature.bin'
    res = array.array('f')
    # print(res)
    fr = open(binary_file, 'rb')
    fr.seek(index_name_array[0][0] * offset)
    # print(fr)
    res.fromfile(fr, dim)
    print(res)
    previous = index_name_array[0][0]
    # print(previous)

    for next in sorted_index[1:]:
        print("enter into")
        move = (next - 1 - previous) * offset
        # print next, move
        fr.seek(move, 1)
        res.fromfile(fr, dim)
        previous = next

    fr.close()
    return [x[1] for x in index_name_array], [res[i * dim:(i + 1) * dim].tolist() for i in range(nr_of_images)]


# check frames in train|val|test dataset  # 总305462帧
# train video_frames number: 6513
# val video_frames number: 497
# test video_frames number: 2990
# shortest frame number: 20
def check_video(phase):
    path = ['/root/VisualSearch/msrvtt10k{}/FeatureData/resnet-152-img1k-flatten0_outputos'.format(x) for x in phase]
    for x in path:
        file_path = x + '/id.txt'
        file = open(file_path).read().strip().split()
        # print(len(file))
        dic = {}
        for item in file:
            video_name = item.split('_')[0]
            if video_name in dic.keys():
                dic[video_name] += 1
            else:
                dic[video_name] = 1
        print(len(dic))
        # minimal = 100
        # for key, value in dic.items():
        #     if value < minimal:
        #         minimal = value
        # for key,value in dic.items():
        #     if value==minimal:
        #         print(key)
        # print("shortest frame number:", minimal)
        # print(dic)


def test_gru():
    # GRU
    rnn = nn.GRU(10, 20, 2)
    input = torch.randn(5, 3, 10)
    h0 = torch.randn(2, 3, 20)
    output, hn = rnn(input, h0)
    print(output)
    print(hn)
    print(output.size(), hn.size())


def deal_with_invalid_frame():
    """
    清理无效帧
    # 有效帧个数508373
    :return:
    """
    video_frames_path = 'video_frames/'
    frame_list = os.listdir(video_frames_path)
    invalid_cnt = 0
    for i, frame in enumerate(frame_list):
        f = cv2.imread(os.path.join(video_frames_path, frame))
        if f is None:
            os.remove(os.path.join(video_frames_path, frame))
            invalid_cnt += 1
        if i % 200 == 0:
            print("process frame ", i)
            print(invalid_cnt)
        i += 1
    print(invalid_cnt)


def check_txt():
    # json以字典格式保存了一类汽车的信息
    # 两个关键字 GraphImages GraphProfileInfo
    json_file = 'dataset/ins_Car_data/abarth_official/abarth_official.json'
    dic = json.load(open(json_file, 'r', encoding='utf-8'))
    print(len(dic))


def change_filck_w2v_encoding():
    path = '/root/VisualSearch/word2vec/flickr/vec500flickr30m/id.txt'
    target = '/root/VisualSearch/word2vec/flickr/vec500flickr30m/id_1.txt'
    items = open(path).read().strip().split()
    items_ = open(target).read().strip().split()
    print(len(items), len(items_))
    # f = codecs.open(target, 'w', encoding='utf8')
    # f.write(" ".join(items).decode('unicode_escape'))
    # print('finished.')
    # f.close()


def bert_test():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # sequence = "A Titan RTX has 24GB of VRAM"
    # tokenized_sequence = tokenizer.tokenize(sequence)
    # # print(tokenized_sequence)
    # sentence1 = "The man was accused of robbing a bank."
    # sentence2 = "The man went fishing by the bank of the river."
    # tokenized_text = tokenizer.tokenize(sentence2)
    # print(tokenized_text)
    # encoded_sequence = tokenizer(sentence2)["input_ids"]
    # print(encoded_sequence)
    # segments_ids = [1]*len(tokenized_text)
    # print(segments_ids)
    # indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # for tup in zip(tokenized_text, indexed_tokens):
    #     print(tup)
    #
    # padded_sequences = tokenizer([sentence1, sentence2, 'haha'], padding=True)
    # print(padded_sequences)
    config = BertConfig(num_hidden_layers=3, num_attention_heads=8)
    print(config)
    model = BertModel.from_pretrained('bert-base-uncased', config=config)
    # print(model)
    batch_sentences = ("Hello I'm a single sentence",
                       "And another sentence",
                       "And the very very last one")
    #  pad to the longest sequence in the batch
    # truncate to a maximum length accepted by the model
    inputs = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
    # print(inputs)
    outputs = model(**inputs, output_hidden_states=True)
    # print(len(outputs[2]))
    # print(outputs[0]) # last hidden state
    # print("*"*50)
    # print(outputs[2][3])
    # print(len(inputs["input_ids"]))
    # print(int(torch.sum(inputs["input_ids"][0])))
    # # last_hidden_state  pooler_output(cls token) hidden_states(type: tuple,one for each layer)
    # # print(len(outputs)) # 3
    # print(len(outputs[2]))
    # # for index in range(len(outputs[2])):
    # #     print(outputs[2][index].shape)
    # print(outputs[0])
    # print(torch.stack(outputs[2], 0)[-4:].sum(0).shape)
    # configuration = BertConfig(num_hidden_layers=3)
    # print(configuration)
    # pretrained_weights = 'bert-base-uncased'
    # model = BertModel.from_pretrained(pretrained_weights, config=configuration)
    # configuration = model.config
    # print(configuration)


def test_tensorboard():
    writer = SummaryWriter()

    for n_iter in range(100):
        writer.add_scalar('Loss/train', np.random.random(), n_iter)
        writer.add_scalar('Loss/test', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/test', np.random.random(), n_iter)


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


def dataset_info():
    brand_path = '/root/VisualSearch/ins_Car_data'
    brand_list = os.listdir(brand_path)
    brand_list.sort()
    for index, brand in enumerate(brand_list):
        cnt = 0
        # print("process brand number:{}/{}".format(index, len(brand_list)))
        files = os.listdir(os.path.join(brand_path, brand))
        files.sort()
        for file in files:
            # 只处理视频
            if not file.endswith("mp4"):
                continue
            cnt += 1
        # brand info
        print('{} videos in brand {}'.format(cnt, index))


if __name__ == '__main__':
    # x = torch.Tensor([[1, 2], [3, 4]])
    # x = l2norm(x)
    # print(x)
    #
    # a = torch.Tensor(32, 2000, 1024)
    # b = a.mean(1)
    # print(b.shape)
    # b = torch.Tensor(1024, 1)
    # c = a.matmul(b)
    # d = c.squeeze()
    # e = d.mean(1)
    # print(c.shape)
    # print(d.shape)
    # print(e.shape)
    # print(e)
    # print(torch.cuda.is_available())
    # scores = torch.empty((32, 32), device='cuda')
    # print(scores.shape)
    # for i in range(a.shape[0]):
    #     e = a.matmul(b).squeeze().mean(1)
    #     scores[i] = e
    # dataset_info()
    # a = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # diagonal = a.diag().view(a.size(0), 1)
    # b = diagonal.t().expand_as(a)
    # # c = diagonal.expand_as(a)
    # I = torch.eye(a.size(0)) > .5
    # print(a.max(0))
    # print(b)
    # # print(c)
    # cost = (0.5 + a - b).clamp(min=0)
    # print(cost)
    # cost = cost.masked_fill_(I, 0)
    # print(cost)
    # print(cost.max(0))
    # x = torch.LongTensor([0,1])
    # y = torch.LongTensor([0,1])
    # print(x[1])
    # print(y[1])
    # print(x[1] == y[1])
    # a = [1,2,3,4,5,6,7,8,9,10]
    # random.seed(2)
    # random.shuffle(a)
    # print(a)
    # a = ['1']
    # if isinstance(a, str):
    #     print('haha')
    # a = [(1,2),(3,4),(5,6)]
    # f = open("loss.txt", 'w')
    # f.write(str(a))
    # a = ['ss']
    # print(' '.join(a))
    # if isinstance(a, str):
    #     print('haha')
    # verticals_file = "verticals/verticals.txt"
    # file = read_dict(verticals_file)
    # print(file.keys())
    # import pickle
    # bow_vocab_file = "/root/VisualSearch/insCar/insCartrain/TextData/vocabulary/bow/word_vocab_5.pkl"
    # bow_vocab = pickle.load(open(bow_vocab_file, 'rb'))
    # print("bow_vocab_size ", len(bow_vocab))
    strs = []
    strs.append("124")
    strs.append("244")
    with open("mytest.txt",'w') as f:
        f.write(str(strs))
