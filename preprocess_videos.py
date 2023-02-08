import cv2
import os
from basic.util import write_dict, read_dict


"""对数据集中的原始视频数据进行处理，利用opencv对视频抽帧
"""

def video2frame(root, videos_path, frames_save_path):
    '''
    :param videos_path: 所有视频的存放路径
    :param frames_save_path: 所有视频切分成帧之后图片的保存路径
    :param time_interval: 取帧的间隔
    :return:
    '''
    # 区分是单个路径或者路径列表
    if isinstance(videos_path, str):
        video_category_list = os.listdir(videos_path)
    else:
        video_category_list = videos_path

    # 视频类别序号按字典序处理
    video_category_list.sort()
    print(video_category_list)
    video_id = 0
    for index, cate in enumerate(video_category_list):
        print("process video:{}/{}".format(index+1, len(video_category_list)))
        files = os.listdir(os.path.join(root, cate))
        files.sort()
        for file in files:
            # 只处理视频
            if not file.endswith("mp4"):
                continue
            video_id += 1
            vidcap = cv2.VideoCapture(os.path.join(root, cate, file))
            # 帧率
            fps = int(round(vidcap.get(cv2.CAP_PROP_FPS)))
            # 抽帧的间隔 每半秒取一帧
            time_interval = fps//2
            # 分辨率[宽度]
            width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
            # 分辨率[高度]
            height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # 总帧数
            frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            print("[fps-width-height-frames]", fps, width, height, frames)
            count = 0
            # 判断是否正常打开
            if vidcap.isOpened():
                reval, frame = vidcap.read()
                print("video_frames read success.")
            else:
                print("open failed.")
                reval = False
            # 输出图片的目录
            if not os.path.exists(frames_save_path):
                print("create output dir.")
                os.makedirs(frames_save_path)
            while reval:
                reval, frame = vidcap.read()
                if count % time_interval == 0:
                    # 帧命名：视频id + 当前帧位置 + 视频类别序号
                    frame_name = "video"+str(video_id)+"_"+str(count)+"_cls"+str(index)+".jpg"
                    output_file_path = os.path.join(frames_save_path, frame_name)
                    print(output_file_path)
                    try:
                        cv2.imwrite(output_file_path, frame)
                    except:
                        pass
                count += 1
            vidcap.release()

# 计算视频到视频id的映射，以及视频id到视频的映射
def video2idx_and_idx2video(root_path, videos_path, vertical):
    if isinstance(videos_path, str):
        video_category_list = os.listdir(videos_path)
    else:
        video_category_list = videos_path
    video_category_list.sort()
    video_id = 0
    video2idx = {}
    idx2video = {}
    duplicate_cnt = 0
    for index, cate in enumerate(video_category_list):
        print("process dir:{}/{}".format(index + 1, len(video_category_list)))
        files = os.listdir(os.path.join(root_path, cate))
        files.sort()
        for file in files:
            # 只处理视频
            if not file.endswith("mp4"):
                continue
            video_id += 1
            video_name = file[:-4]
            print(video_name)
            if video_name not in video2idx.keys():
                video2idx[video_name] = video_id
                idx2video[video_id] = video_name
            else:
                print("duplicate video name：", video_name)
                duplicate_cnt += 1
                # print("exist videos has same name!")
                # exit(1)

    print("total duplicated videos:", duplicate_cnt) # 12
    video_info = {"video2idx": video2idx, "idx2video": idx2video}
    path = "/root/VisualSearch/verticals"
    target_path = os.path.join(path, vertical, "video_info.txt")
    write_dict(target_path, video_info)


if __name__ == '__main__':
    pass
    # 视频集合的根目录
    # 所有视频的输出帧保存目录
    # frames_save_root_path = 'video_frames'

    # for cate in category_list:
    #     if not len(os.listdir(os.path.join(videos_root_path,cate))):
    #         print(cate)
    # video2frame(videos_root_path, frames_save_root_path)
