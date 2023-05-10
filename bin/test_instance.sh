rootpath=/group_homes/public_cluster/home/u190110105/insCar
testCollection=insCartest
logger_name=/group_homes/public_cluster/home/u190110105/insCar/model/$1
n_caption=1
overwrite=1
batch_size=64
#checkpoint_name=checkpoint_epoch_$2.pth.tar

gpu=0
CUDA_VISIBLE_DEVICES=$gpu python ../tester.py $testCollection --rootpath $rootpath --overwrite $overwrite \
          --n_caption $n_caption --batch_size $batch_size \
          --logger_name $logger_name