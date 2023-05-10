rootpath=/Users/pinskyrobin/Downloads/insCar
testCollection=insCartest
logger_name=/Users/pinskyrobin/Downloads/insCar/model/runs_0
n_caption=1
overwrite=1
batch_size=8
checkpoint_name=checkpoint_epoch_$1.pth.tar

python /Users/pinskyrobin/Downloads/Brand/tester.py $testCollection \
--rootpath $rootpath --overwrite $overwrite --logger_name $logger_name \
--n_caption $n_caption --batch_size $batch_size \
--checkpoint_name $checkpoint_name