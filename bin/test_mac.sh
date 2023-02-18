# 生成执行测试过程脚本的模板 可以选择在训练阶段结束时自动测试
rootpath=/Users/pinskyrobin/Downloads/insCar
testCollection=insCartest
logger_name=/Users/pinskyrobin/Downloads/insCar/model
n_caption=1
overwrite=1
batch_size=8
checkpoint_name=checkpoint_epoch_$1.pth.tar

python /Users/pinskyrobin/Downloads/Brand/tester.py $testCollection --rootpath $rootpath --overwrite $overwrite --logger_name $logger_name --n_caption $n_caption --checkpoint_name $checkpoint_name --batch_size $batch_size