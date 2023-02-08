import os
import random

from basic.util import write_dict, read_dict
import json
from util.vocab import clean_str

"""
对数据集中的原始文本captions进行预处理，包括图像的captions和视频的captions
"""
def extract_image_captions(root_path, brand_path, vertical):
    """
    抽取每个image的captions
    :param root_path: 品牌数据的根目录
    :return:
    """
    if isinstance(brand_path, str):
        brand_list = os.listdir(root_path)
    else:
        brand_list = brand_path
    brand_list.sort()
    json_file_cnt = 0
    img2captions = {}
    img_cnt = 0
    for index, cate in enumerate(brand_list):
        print("process dir:{}/{}".format(index + 1, len(brand_list)))
        files = os.listdir(os.path.join(root_path, cate))
        files.sort()
        for file in files:
            # captions保存在json文件中
            if not file.endswith(".json"):
                continue
            json_file_cnt += 1
            dic = json.load(open(os.path.join(root_path, cate, file), 'r', encoding='utf-8'))
            data_items = dic['GraphImages']
            # get each dict
            for item in data_items:
                # 只处理图片数据
                if item['__typename'] == "GraphImage" or not item["is_video"]:
                    # identify mark is "shortcode"
                    image_name = cate + '/' + item["shortcode"] + '.jpg'
                    # skip duplicated video names and null captions
                    if image_name not in img2captions.keys() and len(item["edge_media_to_caption"]["edges"]):
                        caps = item["edge_media_to_caption"]["edges"][0]["node"]["text"]
                        if caps is None:
                            continue
                        # print(len(item["edge_media_to_caption"]["edges"]))
                        tags = item["tags"] if "tags" in item.keys() else None
                        img_cnt += 1
                        img2captions[image_name] = {"caps": caps, "tags": tags}

    print("captions file cnt:", json_file_cnt)  # 50
    print('total img captions {}'.format(len(img2captions)))  # 98259
    caps_dir = "img_captions.txt"
    # save video caps to file
    path = "/root/VisualSearch/verticals"
    caps_dir = os.path.join(path, vertical, caps_dir)
    with open(caps_dir, 'w') as f:
        f.write(json.dumps(img2captions))


def extract_video_captions(root_path, brand_path, vertical):
    """
    抽取每个video的captions
    :param root_path:
    :return:
    """
    if isinstance(brand_path, str):
        video_category_list = os.listdir(root_path)
    else:
        video_category_list = brand_path
    video_category_list.sort()
    json_file_cnt = 0
    cls2idx = {}
    idx2cls = {}
    video2captions = {}
    video_cnt = 0
    for index, cate in enumerate(video_category_list):
        cls_name = cate.split('/')[-1]
        cls2idx[cls_name] = index
        idx2cls[index] = cls_name
        print("process dir:{}/{}".format(index + 1, len(video_category_list)))
        files = os.listdir(os.path.join(root_path, cate))
        files.sort()
        for file in files:
            # captions保存在json文件中
            if not file.endswith(".json"):
                continue
            json_file_cnt += 1
            dic = json.load(open(os.path.join(root_path, cate, file), 'r', encoding='utf-8'))
            data_items = dic['GraphImages']
            # get each dict
            for item in data_items:
                # skip the images, focus on videos
                if item['__typename'] == "GraphVideo" and item["is_video"]:
                    # identify mark is "shortcode"
                    video_name = item["shortcode"]
                    # skip duplicated video names and null captions
                    if video_name not in video2captions.keys() and len(item["edge_media_to_caption"]["edges"]):
                        caps = item["edge_media_to_caption"]["edges"][0]["node"]["text"]
                        # print(len(item["edge_media_to_caption"]["edges"]))
                        if caps is None:
                            continue
                        tags = item["tags"] if "tags" in item.keys() else None
                        video_cnt += 1
                        video2captions[video_name] = {"caps": caps, "tags": tags}

    print("captions file cnt:", json_file_cnt)  # 50
    caps_dir = "video_captions.txt"
    cls_dir = "cls.txt"
    path = "/root/VisualSearch/verticals"
    caps_dir = os.path.join(path, vertical, caps_dir)
    cls_dir = os.path.join(path, vertical, cls_dir)
    cls = {"cls2idx": cls2idx, "idx2cls": idx2cls}

    # save cls info to file
    with open(cls_dir, 'w') as f:
        f.write(json.dumps(cls))
    # save video caps to file
    with open(caps_dir, 'w') as f:
        f.write(json.dumps(video2captions))


def imgs_split_train_val_test(source_root_path, target_root_path, vertical, brand_path):
    """
    将img和对应的caption 划分train/val/test 比例80:5:15
    :return:
    """
    # 排除没有caption的imgs,剩下的数据用来划分
    # 文本包括captions和tag,缺少tag的样本没有被删除
    img_captions_path = os.path.join(target_root_path, vertical, "img_captions.txt")
    img_info_path = os.path.join(target_root_path, vertical, "img_info.txt")
    f = open(img_captions_path, 'r').read()
    caps = json.loads(f)
    # print(caps)
    # get all img names
    # 不带jpg后缀
    img_names = caps.keys()
    # img2idx and idx2img
    img_info = read_dict(img_info_path)
    # 带jpg后缀
    img2id = img_info["img2idx"]
    print(len(img2id))
    id2img = img_info["idx2img"]
    print(len(id2img))

    ids = [img2id[img] for img in img_names if img in img2id]
    # print(ids)
    print('有效图片数', len(ids))

    # 读取品牌数据集
    if isinstance(brand_path, str):
        brand_list = os.listdir(brand_path)
    else:
        brand_list = brand_path
    brand_list.sort()
    items = []
    total_imgs_num = 0
    zero_shot_brand = 0
    train_img_ids = []
    val_img_ids = []
    test_img_ids = []
    for index, brand in enumerate(brand_list):
        print("process brand number:{}/{}".format(index + 1, len(brand_list)))
        files = os.listdir(os.path.join(source_root_path, brand))
        files.sort()
        items.clear()
        for file in files:
            if not file.endswith("jpg"):
                continue
            img = brand + '/' + file
            if img in img2id and img2id[img] in id2img:
                items.append(img2id[img])
        # brand info
        print('{} img points in brand {}'.format(len(items), index))
        if len(items) == 0:
            zero_shot_brand += 1
        # divide img points into train/val/test set randomly
        random.seed(index)
        random.shuffle(items)
        one_piece = len(items) // 20
        train_img_ids.extend(items[:one_piece * 16])
        val_img_ids.extend(items[one_piece * 16:one_piece * 17])
        test_img_ids.extend(items[one_piece * 17:])
        total_imgs_num += len(items)

    print("random split datasets finished..")
    print("total num of data points {}".format(total_imgs_num))
    print("totally {} brands do not have img datas".format(zero_shot_brand))
    print(
        "train_size {} val_size {} test_size {}".format(len(train_img_ids), len(val_img_ids), len(test_img_ids)))
    prefix = vertical
    dataset = {"train": train_img_ids, "val": val_img_ids, "test": test_img_ids}

    # 出错记录条数
    errors = None
    # obtain train/val/test dataSet
    for x in dataset:
        captions_path = prefix + x + ".img_caption.txt"
        captions_path = os.path.join(target_root_path, vertical, captions_path)
        if os.path.exists(captions_path):
            os.remove(captions_path)
        errors = 0
        with open(captions_path, 'a+') as writer:
            for id in dataset[x]:
                try:
                    text = caps[id2img[id]]["caps"]
                except:
                    errors += 1
                    continue
                # sentence + tag 作最终的caption
                text = clean_str(text)
                text = " ".join(text)
                string = "img" + str(id) + "#enc#0" + " " + text + '\n'
                writer.write(string)
        print("error data num is {} in {}".format(errors, x))


def videos_split_train_val_test(source_root_path, target_root_path, vertical, brand_path):
    """
    利用已经提取好的video和caption  划分出train/val/test 尽量使得各品牌的数据量在train/val/test中分布均匀些
    本实验划分比例约 80%:5%:15%
    :return:
    """
    # 排除没有caption的videos,剩下的数据用来划分
    # 文本包括captions和tag,缺少tag的样本没有被删除
    video_captions_path = os.path.join(target_root_path, vertical, "video_captions.txt")
    video_info_path = os.path.join(target_root_path, vertical, "video_info.txt")
    f = open(video_captions_path, 'r').read()
    caps = json.loads(f)
    # get all video names
    video_names = caps.keys()
    # name2idx and idx2name
    video_info = read_dict(video_info_path)
    video2id = video_info["video2idx"]
    # print(len(video2id))
    id2video = video_info["idx2video"]
    # print(len(id2video))
    print(len(video_names))
    print(video2id)
    ids = [video2id[video] for video in video_names if video in video2id]
    print(ids)
    print(len(ids))  # 10861

    # 读取品牌数据集
    if isinstance(brand_path, str):
        brand_list = os.listdir(brand_path)
    else:
        brand_list = brand_path
    brand_list.sort()
    items = []
    total_videos_num = 0
    zero_shot_brand = 0
    train_video_ids = []
    val_video_ids = []
    test_video_ids = []
    for index, brand in enumerate(brand_list):
        print("process brand number:{}/{}".format(index + 1, len(brand_list)))
        files = os.listdir(os.path.join(source_root_path, brand))
        files.sort()
        items.clear()
        for file in files:
            # 只处理视频
            if not file.endswith("mp4"):
                continue
            video = file[:-4]
            if video in video2id and video2id[video] in id2video:
                items.append(video2id[video])
        # brand info
        print('{} data points in brand {}'.format(len(items), index))
        if len(items) == 0:
            zero_shot_brand += 1
        # divide data points into train/val/test set randomly
        random.seed(index)
        random.shuffle(items)
        one_piece = len(items)//20
        train_video_ids.extend(items[:one_piece*16])
        val_video_ids.extend(items[one_piece*16:one_piece*17])
        test_video_ids.extend(items[one_piece*17:])
        total_videos_num += len(items)

    print("random split datasets finished..")
    print("total num of data points {}".format(total_videos_num))
    print("totally {} brands do not have video datas".format(zero_shot_brand))
    print("train_size {} val_size {} test_size {}".format(len(train_video_ids), len(val_video_ids), len(test_video_ids)))
    prefix = vertical
    dataset = {"train": train_video_ids, "val": val_video_ids, "test": test_video_ids}

    # 出错记录条数
    errors = None
    # obtain train/val/test dataSet
    for x in dataset:
        captions_path = prefix + x + ".caption.txt"
        captions_path = os.path.join(target_root_path, vertical, captions_path)
        if os.path.exists(captions_path):
            os.remove(captions_path)
        errors = 0
        with open(captions_path, 'a+') as writer:
            # print(len(dataset[x]))
            for id in dataset[x]:
                try:
                    text = caps[id2video[id]]["caps"]
                except:
                    errors += 1
                    continue
                # sentence + tag 作最终的caption
                text = clean_str(text)
                text = " ".join(text)
                string = "video"+str(id)+"#enc#0" + " " + text + '\n'
                writer.write(string)
        print("error data num is {} in {}".format(errors, x))


def merge_captions_in_videos_and_imgs(target_root_path, vertical):
    """
    合并视频的captions和文本的captions为一个文件
    :return:
    """
    prefix = vertical
    dset = ['train', 'val', 'test']
    postfix = '.img_caption.txt'
    for x in dset:
        source_file = prefix + x + postfix
        source_file = os.path.join(target_root_path, vertical, source_file)
        target_file = prefix + x + '.caption.txt'
        target_file = os.path.join(target_root_path, vertical, target_file)
        f_s = open(source_file, 'r').readlines()
        f_t = open(target_file, 'a+')
        f_t.writelines(f_s)
        f_t.close()


if __name__ == '__main__':
    # generate_train_val_test()
    root_path = '/root/VisualSearch/ins_Car_data'
    # extract_image_captions(root_path)
    # imgs_split_train_val_test()
    merge_captions_in_videos_and_imgs()

