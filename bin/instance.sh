rootpath=/group_homes/public_cluster/home/u190110105/insCar
trainCollection=insCartrain
valCollection=insCarval
testCollection=insCartest
video_feature=resnet152_dim_2048
img_feature=imgfeat_dim_2048
metric=auc
margin=0.2
n_caption=1
overwrite=1
measure=cosine
num_epochs=30
brand_num=51
workers=8
batch_size=8
accumulation_step=8
learning_rate=0.0001
brand_aspect=2000
text_mapping_size=1024
visual_mapping_size=1024
common_embedding_size=1024
fusion_style=ph
loss_fun=cl # mrl|CrossCLR|cl
cost_style=mean # mean|sum
concate=full # full|reduced
text_net=transformers # bi-gru|transformers
postfix=ph_cl_mean_0001_without_prune

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
                                            --max_violation --postfix final \
                                            --measure $measure --cost_style $cost_style \
                                            --brand_aspect $brand_aspect

bash test_instance.sh final
