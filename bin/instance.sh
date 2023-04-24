rootpath=/group_homes/public_cluster/home/u190110105/insCar # experiments on XXXXX dataset 调整前四行确定实验用的数据集
trainCollection=insCartrain
valCollection=insCarval
testCollection=insCartest
video_feature=resnet152_dim_2048  # where the videos feature file saved
img_feature=imgfeat_dim_2048 # where the images feature file saved
loss_fun=cl # mrl|CrossCLR
# 多级特征的拼接方式
concate=full # full|reduced
overwrite=1
num_epochs=100 # use early stopping mechanism
text_net=transformers # bi-gru|transformers
batch_size=8
metric=auc
n_caption=1 # how many captions in each video
learning_rate=0.0001
# text feature dim after processed by text_net
text_mapping_size=1024
# visual feature dim after processed by visual_net
visual_mapping_size=1024
# final dim in common space
common_embedding_size=1024
margin=0.2
# final fusion style of Visual and Text
fusion_style=ph
workers=8
brand_num=51
postfix=instance_ph_0001_without_glb
measure=cosine
accumulation_step=8
visual_kernel_sizes=3-5
brand_aspect=512

gpu=0
CUDA_VISIBLE_DEVICES=$gpu python ../trainer.py $trainCollection $valCollection $testCollection \
                                            --rootpath $rootpath \
                                            --workers $workers \
                                            --brand_num $brand_num \
                                            --overwrite $overwrite \
                                            --text_norm --visual_norm \
                                            --video_feature $video_feature --img_feature $img_feature \
                                            --n_caption $n_caption --concate $concate --loss_fun $loss_fun \
                                            --num_epochs $num_epochs --text_net $text_net --batch_size $batch_size \
                                            --accumulation_step $accumulation_step \
                                            --metric $metric --learning_rate $learning_rate \
                                            --common_embedding_size $common_embedding_size \
                                            --text_mapping_size $text_mapping_size \
                                            --visual_mapping_size $visual_mapping_size --margin $margin \
                                            --fusion_style $fusion_style \
                                            --max_violation --postfix $postfix \
                                            --measure $measure \
                                            --brand_aspect $brand_aspect
#                                            --visual_kernel_sizes $visual_kernel_sizes \

#gpu=0
#CUDA_VISIBLE_DEVICES=$gpu nohup python -u ../trainer.py $trainCollection $valCollection $testCollection \
#                                            --rootpath $rootpath \
#                                            --workers $workers \
#                                            --brand_num $brand_num \
#                                            --overwrite $overwrite \
#                                            --text_norm --visual_norm \
#                                            --video_feature $video_feature --img_feature $img_feature \
#                                            --n_caption $n_caption --concate $concate --loss_fun $loss_fun \
#                                            --num_epochs $num_epochs --text_net $text_net --batch_size $batch_size \
#                                            --accumulation_step $accumulation_step \
#                                            --metric $metric --learning_rate $learning_rate \
#                                            --common_embedding_size $common_embedding_size \
#                                            --text_mapping_size $text_mapping_size \
#                                            --visual_mapping_size $visual_mapping_size --margin $margin \
#                                            --fusion_style $fusion_style \
#                                            --max_violation --postfix $postfix \
#                                            --measure $measure --visual_kernel_sizes $visual_kernel_sizes \
#                                            --brand_aspect $brand_aspect
#                                            > "/group_homes/public_cluster/home/u190110105/FGMCD/bg_training.log" 2>&1