rootpath=/root/pinskyrobin/mini_insCar/@@@vertical@@@
collection=@@@collection@@@
# msrvtt10ktrain
threshold=5
overwrite=1
for text_style in bow rnn
do
/root/anaconda3/bin/python ../preprocess/vocab.py $collection --rootpath $rootpath --threshold $threshold --text_style $text_style --overwrite $overwrite
done