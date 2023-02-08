'''
# convert one or multiple feature files from txt format to binary (float32) format
'''

import os, sys, math
import numpy as np
from optparse import OptionParser


def checkToSkip(filename, overwrite):
    if os.path.exists(filename):
        print("%s exists." % filename),
        if overwrite:
            print("overwrite")
            return 0
        else:
            print("skip")
            return 1
    return 0


def process(feat_dim, inputTextFiles, resultdir, overwrite ):
    """
    :param feat_dim: 每帧图像特征的维度/每张图像的特征维度
    :param inputTextFiles: 存放特征的文件或目录
    :param resultdir:  保存二进制格式特征的位置
    :param overwrite:  是否覆盖二进制特征文件
    :return:
    """
    res_binary_file = os.path.join(resultdir, 'feature.bin')
    res_id_file = os.path.join(resultdir, 'id.txt')

    if checkToSkip(res_binary_file, overwrite):
        return 0

    if os.path.isdir(resultdir) is False:
        os.makedirs(resultdir)

    fw = open(res_binary_file, 'wb')
    processed = set()
    imset = []
    count_line = 0
    failed = 0

    # 扫描所有存放特征的目录
    for filename in inputTextFiles:
        print('>>> Processing %s' % filename)
        filename = filename.strip()
        #   一行对应为一帧的名字 + 帧特征 2048维
        lines = open(filename).readlines()

        for line in lines:
            count_line += 1
            if count_line % 100 == 0:
                print("processing progress:{}/{}".format(count_line, len(lines)))
            elems = line.strip().split()
            # 如果处理的是图像特征 注意图像名中可能包含空格 按空格切分后就不能只是取第一项做图像名
            # 那么按照原先约定的维度 确定特征值的长度 剩下的拼接成图像名
            if not elems:
                continue
            # name = elems[0]
            feature_values = elems[-feat_dim:]
            name = " ".join(elems[:len(elems)-feat_dim])
            if name in processed:
                continue
            processed.add(name)

            # del elems[0]
            # 特征向量转浮点型
            try:
                vec = np.array(feature_values, dtype=np.float32)
            except:
                print(elems)
                break
            # print(vec)
            okay = True
            for x in vec:
                if math.isnan(x):
                    okay = False
                    break
            if not okay:
                failed += 1
                continue
          
            if feat_dim == 0:
                feat_dim = len(vec)
            else:
                assert(len(vec) == feat_dim), "dimensionality mismatch: required %d, input %d, id=%s, inputfile=%s" % (feat_dim, len(vec), name, filename)
            vec.tofile(fw)
            # print name, vec
            imset.append(name)
    fw.close()
    # if os.path.exists(res_id_file):
    #     os.remove(res_id_file)
    fw = open(res_id_file, 'w', encoding='utf-8')
    """
    图像名和视频帧名 均以#界定
    """
    fw.write('#'.join(imset))
    # 处理视频时视频帧的名字以' '界定
    # fw.write(' '.join(imset))
    fw.close()
    fw = open(os.path.join(resultdir, 'shape.txt'), 'w')
    fw.write('%d %d' % (len(imset), feat_dim))
    fw.close() 
    print ('%d lines parsed, %d ids,  %d failed ->  %d unique ids' % (count_line, len(processed), failed, len(imset)))


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = OptionParser(usage="""usage: %prog [options] nDims inputTextFile isFileList resultDir""")
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default=0)")

    (options, args) = parser.parse_args(argv)
    if len(args) < 4:
        parser.print_help()
        return 1

    fea_dim = int(args[0])
    inputTextFile = args[1]
    print(fea_dim)
    print(inputTextFile)
    # 判断要处理的特征文件是不是目录
    if int(args[2]) == 1:
        # 依次读每个目录里的特征
        inputTextFiles = [x.strip() for x in open(inputTextFile).readlines() if x.strip() and not x.strip().startswith('#')]
    # 若只是单个文件
    else:
        inputTextFiles = [inputTextFile]
    return process(fea_dim, inputTextFiles, args[3], options.overwrite)


if __name__ == "__main__":
    sys.exit(main())

