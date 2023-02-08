# 构建训练/验证/测试目录
cd /root/VisualSearch/verticals/@@@vertical@@@
mkdir -p @@@vertical@@@train/FeatureData
mkdir -p @@@vertical@@@val/FeatureData
mkdir -p @@@vertical@@@test/FeatureData

mkdir -p @@@vertical@@@train/TextData/
mkdir -p @@@vertical@@@val/TextData/
mkdir -p @@@vertical@@@test/TextData/

cp -r resnet152_dim_2048/ @@@vertical@@@train/FeatureData/
cp -r imgfeat_dim_2048/ @@@vertical@@@train/FeatureData/
cp -r resnet152_dim_2048/ @@@vertical@@@val/FeatureData/
cp -r imgfeat_dim_2048/ @@@vertical@@@val/FeatureData/
cp -r resnet152_dim_2048/ @@@vertical@@@test/FeatureData/
cp -r imgfeat_dim_2048/ @@@vertical@@@test/FeatureData/

cp @@@vertical@@@train.caption.txt @@@vertical@@@train/TextData/
cp @@@vertical@@@val.caption.txt @@@vertical@@@val/TextData/
cp @@@vertical@@@test.caption.txt @@@vertical@@@test/TextData/
cp -r /root/VisualSearch/word2vec ./