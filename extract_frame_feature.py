"""
视频数据抽帧的特征并保存
"""
from resnet152 import ResNet152
import torchvision.models as models
from mydataset import VideoDataSet
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import torch
import os


def extract(frames_root_path, feat_save_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    # import torch
    # torch.backends.cudnn.enabled = False
    resnet_152 = models.resnet152(pretrained=True)
    model = ResNet152(resnet_152)
    root_path = frames_root_path
    dataset = VideoDataSet(root_path)
    batch_size = 32
    train_loader = DataLoader(dataset,
                              batch_size=batch_size, shuffle=False)
    use_gpu = torch.cuda.is_available()
    print("use_gpu", use_gpu)
    if use_gpu:
        model = model.cuda()

    # 抽取视频特征并保存为txt
    target_feature_file = os.path.join(feat_save_path, 'frame_feature_dim_2048.txt')
    # 文件存在就删除
    if os.path.exists(target_feature_file):
        os.remove(target_feature_file)
    ndim = 2048
    f = open(target_feature_file, 'a+')
    for i, input in enumerate(train_loader):
        print("processing batch {}/{}".format(i + 1, len(dataset)//batch_size))
        feature = input[0]
        frame_names = input[1]
        # print(frame_names)
        if use_gpu:
            feature = feature.cuda()
        feature = torch.squeeze(model(feature)).detach().cpu().numpy()
        # print(feature.shape)
        feature_list = []
        for index in range(feature.shape[0]):
            feature_list.clear()
            feature_list.append(frame_names[index])
            for j in range(ndim):
                feature_list.append(str(feature[index][j]))
            line = " ".join(feature_list)
            f.write(line + '\n')
        # break
    f.close()


if __name__ == '__main__':
    pass
