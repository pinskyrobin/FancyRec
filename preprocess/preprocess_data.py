import pandas as pd
from util.util import read_dict, write_dict
import os
from extract_frame_feature import extract
from txt2bin import process
from get_frameInfo import get_frame_info
from preprocess_videos import video2frame, video2idx_and_idx2video
from preprocess_images import obtain_images, extract_images_features, img2idx_and_idx2img
from preprocess_captions import extract_image_captions, extract_video_captions, imgs_split_train_val_test, \
    videos_split_train_val_test, merge_captions_in_videos_and_imgs


def get_verticals(data_file_path="/root/subbrand_dataset/label.csv"):
    """
    现有数据包含10个垂直领域
    :return:
    """
    data_points = pd.read_csv(data_file_path).values
    # print(len(data_points)) # 193
    # print(data_points[0]) # ['auto' 'bmw' 'bmw' 0 0 0]
    # print(type(data_points[0][3])) # int
    verticals = {}
    for item in data_points:
        if item[0] not in verticals.keys():
            verticals[item[0]] = []
        # 每个领域所包含的品牌
        verticals[item[0]].append(item[2])
    verticals_file = "verticals.txt"
    write_dict(verticals_file, verticals)


def get_datas_from_each_vertical(get_video=True, get_image=True,
                                 _vertical='insCar', dataset_name='insCar',
                                 target_root_path='/root/pinskyrobin/',
                                 source_root_path='/root/brand/ins_Car_data'):
    """
    整理出每个垂直领域的数据及特征 保存到相应目录
    :return:
    """
    verticals = read_dict(os.path.abspath(os.path.join(os.getcwd(), "..", "out", "cls.txt")))
    script_path = os.path.abspath(os.path.join(os.getcwd(), "..", "bin"))

    vertical_list = [key for key in verticals.keys()]
    # print(verticals.keys())
    print(vertical_list)
    # for index in range(2, len(vertical_list)):
    vertical = vertical_list[0]
    # brands = verticals[vertical]
    brands = list(verticals[vertical].keys())

    if get_video:
        ###################################################################################
        """
        一、视频切分成帧
        """
        # 原视频目录
        videos_path = [os.path.join(source_root_path, brand) for brand in brands]
        # 帧数据存放目录
        frames_save_path = os.path.join(target_root_path, dataset_name, "frames")
        # 切帧
        video2frame(source_root_path, videos_path, frames_save_path)
        # 视频与id的映射
        video2idx_and_idx2video(source_root_path, videos_path, _vertical)
        # 将无效帧剔除
        for index, frame_name in enumerate(os.listdir(frames_save_path)):
            print('检查第{}个帧'.format(index))
            file = os.path.join(frames_save_path, frame_name)
            if os.path.isfile(file) and os.path.getsize(file) == 0:  # 如果空文件
                os.remove(file)  # 删除这个文件
        # ####################################################################################
        """
        二、提取视频特征
        """
        # 特征保存目录
        frame_feat_save_path = os.path.join(target_root_path, dataset_name)
        # 特征保存格式txt
        extract(frames_save_path, frame_feat_save_path)
        # ####################################################################################
        """
        三、视频特征整理
        """
        # 特征格式txt转二进制
        feat_dim = 2048
        input_text_file = [os.path.join(frame_feat_save_path, 'frame_feature_dim_2048.txt')]
        bin_feat_dir = os.path.join(frame_feat_save_path, 'resnet152_dim_2048')
        over_write = 1
        process(feat_dim, input_text_file, bin_feat_dir, over_write)
        # 获取视频id到帧的映射
        get_frame_info(bin_feat_dir, over_write)

    if get_image:
        ####################################################################################
        """
        四、提取图像特征
        """
        # 读取品牌的图像集
        images_path = [os.path.join(source_root_path, brand) for brand in brands]
        images_list = obtain_images(source_root_path, images_path, threshold=200)
        # # 图像特征保存的目录
        img_feat_save_path = os.path.join(target_root_path, dataset_name)
        extract_images_features(source_root_path, images_list, img_feat_save_path)
        # ####################################################################################
        """
        五、图像特征整理
        """
        # 特征格式txt转二进制
        feat_dim = 2048
        input_text_file = [os.path.join(img_feat_save_path, 'images_feature_dim_2048.txt')]
        bin_feat_dir = os.path.join(img_feat_save_path, 'imgfeat_dim_2048')
        over_write = 1
        process(feat_dim, input_text_file, bin_feat_dir, over_write)
        # 获取图像名到id的映射
        img2idx_and_idx2img(source_root_path, images_path, _vertical)

    ####################################################################################
    """
    六、获取视频和图像的captions
    """
    if get_video:
        extract_video_captions(source_root_path, videos_path, _vertical)
    if get_image:
        extract_image_captions(source_root_path, images_path, _vertical)
    # ####################################################################################
    """

    七、所有数据划分train/val/test
    """
    # 划分视频和相应captions
    if get_video:
        videos_split_train_val_test(source_root_path, target_root_path, dataset_name, videos_path)
    # 划分图像和相应的captions
    if get_image:
        imgs_split_train_val_test(source_root_path, target_root_path, dataset_name, images_path, threshold=200)
    # 视觉模态、文本模态相应数据合并
    merge_captions_in_videos_and_imgs(target_root_path, dataset_name)
    ####################################################################################
    """
    八、生成数据集的词文件 (用于one-hot)
    """
    commandStr = ''.join(
        open(os.path.join(script_path, 'template_do_get_vertical_vocab.sh'), 'r', encoding='utf8').readlines())
    commandStr = commandStr.replace("@@@vertical@@@", _vertical)
    commandStr = commandStr.replace("@@@collection@@@", _vertical + 'train')
    commandStr = commandStr.replace("@@@dataset_name@@@", dataset_name)
    print(commandStr)
    file = os.path.join(script_path, 'do_get_vertical_vocab.sh')
    open(file, 'w', encoding='utf8').write(commandStr + '\n')
    os.system("chmod +x %s" % file)
    os.system('sh ' + file)
    # ####################################################################################
    """
    九、最终整理
    """
    commandStr = ''.join(open(os.path.join(script_path, "template_construct_dir.sh"), 'r', encoding='utf8').readlines())
    commandStr = commandStr.replace("@@@vertical@@@", _vertical)
    commandStr = commandStr.replace("@@@dataset_name@@@", dataset_name)
    print(commandStr)
    runfile = os.path.join(script_path, "construct_train_val_test_dir.sh")
    open(runfile, 'w', encoding='utf8').write(commandStr + '\n')
    os.system("chmod +x %s" % runfile)
    os.system('sh ' + runfile)
    ####################################################################################


if __name__ == '__main__':
    video = True
    image = True
    get_datas_from_each_vertical(video, image,
                                 _vertical='insCar', dataset_name='insCar_ViT',
                                 source_root_path='/data1/data/brand',
                                 target_root_path='/data1/data/brand')
