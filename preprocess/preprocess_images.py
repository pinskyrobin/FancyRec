import torch
import os

from util.constant import device
from util.util import write_dict
from preprocess.resnet152 import ResNet152
import torchvision.models as models
from mydataset import ImageDataSet
from torch.utils.data import DataLoader

"""对数据集中的原始图像数据进行预处理
"""
def img2idx_and_idx2img(source_root, path, vertical):
    if isinstance(path, str):
        brand_list = os.listdir(path)
    else:
        brand_list = path
    brand_list.sort()
    img_id = 0
    img2idx = {}
    idx2img = {}
    duplicate_cnt = 0
    for index, cate in enumerate(brand_list):
        print("process dir:{}/{}".format(index + 1, len(brand_list)))
        files = os.listdir(os.path.join(source_root, cate))
        files.sort()
        for file in files:
            if not file.endswith("jpg"):
                continue
            img_id += 1
            img_name = cate + '/' + file
            if img_name not in img2idx.keys():
                img2idx[img_name] = img_id
                idx2img[img_id] = img_name
            else:
                print("duplicate img name：", img_name)
                duplicate_cnt += 1
                # print("exist videos has same name!")
                # exit(1)

    print("total duplicated imgs:", duplicate_cnt)
    img_info = {"img2idx": img2idx, "idx2img": idx2img}
    pa = "/root/VisualSearch/verticals"
    target_path = os.path.join(pa, vertical, "img_info.txt")
    write_dict(target_path, img_info)


def obtain_images(root, brand_path):
    # 读取品牌数据集
    if isinstance(brand_path, str):
        brand_list = os.listdir(brand_path)
    else:
        brand_list = brand_path
    brand_list.sort()
    images_list = []
    for index, brand in enumerate(brand_list):
        print("process brand number:{}/{}".format(index + 1, len(brand_list)))
        files = os.listdir(os.path.join(root, brand))
        files.sort()
        image_name = []
        for file in files:
            # 获取图像
            if not file.endswith('jpg'):
                continue
            image_name.append(brand + '/' + file)
        print('{} images in brand {} totally.'.format(len(image_name), index))
        images_list.extend(image_name)
    print("{} images in all".format(len(images_list)))  # 104312
    return images_list


def extract_images_features(root_path, image_name_list, img_feat_save_path):
    """传入品牌数据根目录和图像文件名进行图像特征抽取
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    resnet_152 = models.resnet152(pretrained=True)
    model = ResNet152(resnet_152).eval().to(device)
    dataset = ImageDataSet(root_path, image_name_list)
    batch_size = 64
    train_loader = DataLoader(dataset,
                              batch_size=64, shuffle=False)

    # 抽取图像特征并保存为txt
    target_feature_file = os.path.join(img_feat_save_path, 'images_feature_dim_2048.txt')
    # 文件存在就删除
    if os.path.exists(target_feature_file):
        os.remove(target_feature_file)
    ndim = 2048
    f = open(target_feature_file, 'a+')
    for i, input in enumerate(train_loader):
        print("processing batch {}/{}".format(i+1, len(dataset)//batch_size))
        feature = input[0].to(device)
        img_names = input[1]
        feature = torch.squeeze(model(feature)).detach().cpu().numpy()
        # print(feature.shape)
        # 每行为一张图片的特征 格式为 图片名字 + 2048-dim 特征
        feature_list = []
        for index in range(feature.shape[0]):
            feature_list.clear()
            feature_list.append(img_names[index])
            for j in range(ndim):
                feature_list.append(str(feature[index][j]))
            line = " ".join(feature_list)
            f.write(line + '\n')
        # break
    f.close()


if __name__ == '__main__':
    # image_names_list = obtain_images()
    root_path = '/root/VisualSearch/ins_Car_data'
    img2idx_and_idx2img()
    # extract_images_features(root_path, image_names_list)


