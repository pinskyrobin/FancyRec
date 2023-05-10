ndim=2048
inputTxtFile=frame_feature_dim_2048.txt
isFileList=0
resultDir=resnet152_dim_2048
overwrite=1
python util/txt2bin.py $ndim $inputTxtFile $isFileList $resultDir --overwrite $overwrite