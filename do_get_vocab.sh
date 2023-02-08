collection=insCartrain
# msrvtt10ktrain
threshold=5
overwrite=1
for text_style in bow rnn   # 直接就是名字
do
python util/vocab.py $collection --threshold $threshold --text_style $text_style --overwrite $overwrite 
done