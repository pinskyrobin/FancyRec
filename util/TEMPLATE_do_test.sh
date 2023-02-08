# 生成执行测试过程脚本的模板 可以选择在训练阶段结束时自动测试
rootpath=@@@rootpath@@@
testCollection=@@@testCollection@@@
logger_name=@@@logger_name@@@
n_caption=@@@n_caption@@@
overwrite=@@@overwrite@@@

gpu=2

CUDA_VISIBLE_DEVICES=$gpu python tester.py $testCollection --rootpath $rootpath --overwrite $overwrite --logger_name $logger_name --n_caption $n_caption
