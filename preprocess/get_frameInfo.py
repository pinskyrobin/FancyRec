import os
import sys
from util.constant import ROOT_PATH
from util.common import checkToSkip, makedirsforfile
from util.imgbigfile import ImageBigFile


def read_dict(filepath):
    f = open(filepath, 'r')
    a = f.read()
    dict_data = eval(a)
    f.close()
    return dict_data


def write_dict(filepath, dict_data):
    f = open(filepath, 'w')
    f.write(str(dict_data))
    f.close()


def get_frame_info(feature_dir, overwrite):
    """
    把特征做成二进制格式后才能调用这个函数
    """
    target_result_file = os.path.join(feature_dir, "video2frames.txt")
    if checkToSkip(target_result_file, overwrite):
        sys.exit(0)
    makedirsforfile(target_result_file)
    # print('aaa')
    feat_data = ImageBigFile(feature_dir)
    video2frame_no = {}
    video2cls = {}
    for index, frame_id in enumerate(feat_data.names):
        if index % 200 == 0:
            print("process progress:", index)
        data = frame_id.strip().split("_")
        # print(data)
        video_id = data[0]
        # print(video_id)
        # 帧序号
        fm_no = int(data[1])
        # 视频类别
        video_cls = data[2]
        # print(fm_no)
        video2frame_no.setdefault(video_id, []).append(fm_no)
        if video_id not in video2cls.keys():
            video2cls[video_id] = video_cls
        # if int(fm_no) not in int2str:
        #     int2str[int(fm_no)] = fm_no

    print("save to file start..")
    video2frames = {}
    for video_id, fmnos in video2frame_no.items():
        # 由于存储帧特征时是乱序 取出来重新按帧的时间序排一下
        for fm_no in sorted(fmnos):
            video2frames.setdefault(video_id, []).append(video_id + "_" + str(fm_no) + "_" + video2cls[video_id])

    write_dict(target_result_file, video2frames)
    print("write out into: ", target_result_file)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options]""")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    parser.add_option("--collection", default="", type="string", help="collection name")
    parser.add_option("--feature_dir", default="", type="string", help="feature name")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default=0)")

    (options, args) = parser.parse_args(argv)
    print(options.feature_dir)
    print(options.overwrite)
    return get_frame_info(options.feature_dir, options.overwrite)


if __name__ == "__main__":
    dir = '/home/Brand/resnet152_dim_2048'
    overwrite = 1
    get_frame_info(dir, 1)

