rootpath=/home/u190110105/insCar # experiments on XXXXX dataset 调整前四行确定实验用的数据集
trainCollection=insCartrain
valCollection=insCarval
testCollection=insCartest
video_feature=resnet152_dim_2048  # where the videos feature file saved
img_feature=imgfeat_dim_2048 # where the images feature file saved
loss_fun=mrl
# 多级特征的拼接方式
concate=full # full|reduced
overwrite=1
num_epochs=100 # use early stopping mechanism
text_net=bi-gru # bi-gru|transformer
batch_size=64
metric=auc
n_caption=1 # how many captions in each video
learning_rate=0.0001
# text feature dim after processed by text_net
text_mapping_size=2048
# visual feature dim after processed by visual_net
visual_mapping_size=2048
# final dim in common space
common_embedding_size=1024
margin=0.2
# final fusion style of Visual and Text
fusion_style=fc
workers=0
brand_num=52
#resume=/root/VisualSearch/insCartrain/dinner_project/insCarval/dual_encoding_concate_full_dp_0.2_measure_cosine/vocab_word_vocab_5_word_dim_500_text_rnn_size_512_text_norm_True_text_net_bi-gru_kernel_sizes_2-3-4_num_512/visual_feature_resnet152_img_2048_visual_rnn_size_1024_visual_norm_True_kernel_sizes_2-3-4-5_num_512/mapping_text_1024_img_1024/loss_func_eet_margin_0.3_direction_all_max_violation_True_cost_style_sum/optimizer_adam_lr_0.001_decay_0.99_grad_clip_2.0_val_metric_auc/runs_0/model_best.pth.tar

# Generate a vocabulary on the training set
# ./do_get_vocab.sh $trainCollection
# Generate video_frames frame info
# ./do_get_frameInfo.sh $trainCollection $visual_feature

cd FGMCD/
# training
gpu=-1
CUDA_VISIBLE_DEVICES=$gpu ../anaconda3/bin/python3 trainer.py $trainCollection $valCollection $testCollection \
                                            --rootpath $rootpath \
                                            --workers $workers \
                                            --brand_num $brand_num \
                                            --overwrite $overwrite \
                                            --text_norm --visual_norm \
                                            --video_feature $video_feature --img_feature $img_feature \
                                            --n_caption $n_caption --concate $concate --loss_fun $loss_fun \
                                            --num_epochs $num_epochs --text_net $text_net --batch_size $batch_size \
                                            --metric $metric --learning_rate $learning_rate \
                                            --common_embedding_size $common_embedding_size \
                                            --text_mapping_size $text_mapping_size \
                                            --visual_mapping_size $visual_mapping_size --margin $margin \
                                            --fusion_style $fusion_style \
                                            --max_violation  \
                                            --single_modal_visual  # use visual only
# evaluation (Notice that a script file do_test_${testCollection}.sh will be automatically generated when the training process ends.)
#./do_test_dual_encoding_${testCollection}.sh