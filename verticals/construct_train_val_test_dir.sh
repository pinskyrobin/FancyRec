# 构建训练/验证/测试目录
cd /root/VisualSearch/verticals/alcohol
mkdir -p alcoholtrain/FeatureData
mkdir -p alcoholval/FeatureData
mkdir -p alcoholtest/FeatureData

mkdir -p alcoholtrain/TextData/
mkdir -p alcoholval/TextData/
mkdir -p alcoholtest/TextData/

cp -r resnet152_dim_2048/ alcoholtrain/FeatureData/
cp -r imgfeat_dim_2048/ alcoholtrain/FeatureData/
cp -r resnet152_dim_2048/ alcoholval/FeatureData/
cp -r imgfeat_dim_2048/ alcoholval/FeatureData/
cp -r resnet152_dim_2048/ alcoholtest/FeatureData/
cp -r imgfeat_dim_2048/ alcoholtest/FeatureData/

cp alcoholtrain.caption.txt alcoholtrain/TextData/
cp alcoholval.caption.txt alcoholval/TextData/
cp alcoholtest.caption.txt alcoholtest/TextData/
cp -r /root/VisualSearch/word2vec ./
