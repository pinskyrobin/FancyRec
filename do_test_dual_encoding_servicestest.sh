# 生成执行测试过程脚本的模板 可以选择在训练阶段结束时自动测试
rootpath=/root/VisualSearch/verticals/services
testCollection=servicestest
logger_name=/root/VisualSearch/verticals/services/servicestrain/dinner_project/servicesval/dual_encoding_concate_full_dp_0.2_measure_cosine/vocab_word_vocab_5_word_dim_500_text_rnn_size_512_text_norm_True_text_net_bi-gru_kernel_sizes_2-3-4_num_512/video_feature_resnet152_dim_2048_img_feature_imgfeat_dim_2048_visual_rnn_size_1024_visual_norm_True_kernel_sizes_2-3-4-5_num_512/mapping_text_2048_img_2048/loss_func_mrl_margin_0.2_direction_all_max_violation_True_cost_style_sum/optimizer_adam_lr_0.0001_decay_0.99_grad_clip_2.0_val_metric_auc/runs_0
n_caption=1
overwrite=1

gpu=2

CUDA_VISIBLE_DEVICES=$gpu python tester.py $testCollection --rootpath $rootpath --overwrite $overwrite --logger_name $logger_name --n_caption $n_caption

