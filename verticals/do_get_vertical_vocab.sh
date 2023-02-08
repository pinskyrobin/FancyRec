rootpath=/root/VisualSearch/verticals/alcohol
collection=alcoholtrain
# msrvtt10ktrain
threshold=5
overwrite=1
for text_style in bow rnn   # 直接就是名字
do
/root/anaconda3/bin/python ../util/vocab.py $collection --rootpath $rootpath --threshold $threshold --text_style $text_style --overwrite $overwrite
done
