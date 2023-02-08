# ./do_all.sh msrvtt10ktrain msrvtt10kval msrvtt10ktest full

trainCollection=msrvtt10ktrain
valCollection=msrvtt10kval
testCollection=msrvtt10ktest
concate=full
overwrite=1
gpu=0
# Generate a vocabulary on the training set
#./do_get_vocab.sh $trainCollection

# training

CUDA_VISIBLE_DEVICES=$gpu
python trainer.py $trainCollection $valCollection $testCollection  --overwrite $overwrite \
                                            --max_violation --text_norm --visual_norm --concate $concate


    
# evaluation (Notice that a script file do_test_${testCollection}.sh will be automatically generated when the training process ends.)
# ./do_test_dual_encoding_${testCollection}.sh


