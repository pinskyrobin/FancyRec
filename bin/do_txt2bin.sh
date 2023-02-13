ndim=2048
inputTxtFile=frame_feature_dim_2048.txt
isFileList=0 #特征是单个文件还是一个目录 0代表当个文件
resultDir=resnet152_dim_2048 # 存放输出的二进制特征的位置
overwrite=1
python util/txt2bin.py $ndim $inputTxtFile $isFileList $resultDir --overwrite $overwrite