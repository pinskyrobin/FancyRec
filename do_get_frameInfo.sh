collection=insCartrain  #以自定义数据集的训练集为例
feature_dir=/home/Brand/resnet152_dim_2048
overwrite=1
python util/get_frameInfo.py  --collection $collection --feature_dir $feature_dir --overwrite $overwrite