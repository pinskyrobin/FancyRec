#rootpath=/root/pinskyrobin/insCar/@@@vertical@@@
rootpath=/data1/data/brand/@@@dataset_name@@@/@@@vertical@@@
collection=@@@collection@@@
threshold=5
overwrite=1
for text_style in bow rnn
do
/root/anaconda3/bin/python ../preprocess/vocab.py $collection --rootpath $rootpath --threshold $threshold --text_style $text_style --overwrite $overwrite
done