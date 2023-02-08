import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os

"""
dataset的构造
"""
class ImageDataSet(Dataset):
    """
    construct images dataset.
    """
    def __init__(self, root_path, images_name_list):
        self.root_path = root_path
        self.img_list = images_name_list
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]
        self.transform = transforms.Compose([transforms.Resize((224, 224)),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=self.image_normalization_mean,
                                                                  std=self.image_normalization_std)])

    def __getitem__(self, index):
        img_name = self.img_list[index]
        return self.get(img_name)

    def __len__(self):
        return len(self.img_list)

    def get(self, img_name):
        # 所在目录和文件名
        img_path = os.path.join(self.root_path, img_name)
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, img_name


class VideoDataSet(Dataset):
    """construct video dataset"""
    def __init__(self, root_path):
        self.root_path = root_path
        self.img_list = os.listdir(root_path)
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]
        self.transform = transforms.Compose([transforms.Resize((224, 224)),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=self.image_normalization_mean,
                                                                  std=self.image_normalization_std)])

    def __getitem__(self, index):
        img_name = self.img_list[index]
        return self.get(img_name)

    def __len__(self):
        return len(self.img_list)

    def get(self, img_name):
        img_path = os.path.join(self.root_path, img_name)
        # label = img_name.split('_')[-1][3:]
        frame_name = img_name.split('.')[0]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, frame_name
