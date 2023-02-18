# 生成执行测试过程脚本的模板 可以选择在训练阶段结束时自动测试
rootpath=/home/u190110105/insCar
testCollection=insCartest
logger_name=/home/u190110105/insCar/model
n_caption=1
overwrite=1
checkpoint_name=checkpoint_epoch_$1.pth.tar
batch_size=8

gpu=0
CUDA_VISIBLE_DEVICES=$gpu python /group_homes/public_cluster/home/u190110105/FGMCD/tester.py $testCollection --rootpath $rootpath --overwrite $overwrite --logger_name $logger_name --n_caption $n_caption --checkpoint_name $checkpoint_name --batch_size $batch_size
