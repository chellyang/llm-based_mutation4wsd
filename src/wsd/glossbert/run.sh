#!/bin/bash

conda activate glossbert

cd download/GlossBERT

# extract vocabulary information
# it also prepares glossbert's own datasets

if [ -d wordnet ]
then
echo "GlossBERT prepartion has been done"
else
bash preparation.sh
fi

# prepare my datasets for tasks
echo "Prepare my own evaluation datasets"
python ../../src/wsd/glossbert/preparation.py

#mutation_types="plural gender tense negative"
#mutation_types="antonym comparative demonstrative number passivity that_this inversion tenseplus modifier"
mutation_types="that_this inversion tenseplus modifier"
# create result directory
result_dir="../../result/predictions/glossbert"

if [ -d ${result_dir} ]
then
echo ${result_dir}
else
mkdir -p ${result_dir}
fi

# read argument
read -rp "Enter CUDA ID (eg, 0, 0,1,2): " cuda_id
read -rp "Enter dataset name (multiple datasets seperated by space, eg, ALL):" datasets

# test script
# 显存小的话修改eval_batch_size
for dataset in $datasets
do
    file_prefix="../../asset/Evaluation_Datasets/${dataset}/${dataset}"
    echo ${file_prefix}

    CUDA_VISIBLE_DEVICES=${cuda_id} python run_classifier_WSD_sent.py \
    --task_name WSD \
    --eval_data_dir ${file_prefix}_test_sent_cls_ws.csv \
    --output_dir results \
    --bert_model results/Sent_CLS_WS \
    --do_test \
    --do_lower_case \
    --max_seq_length 512 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --num_train_epochs 6.0 \
    --seed 1314

    # convert output file as gold format
    python ../../src/wsd/glossbert/convert.py \
            --dataset ${dataset} \
            --input_file results/results.txt \
            --output_dir ${result_dir}  

    for mutation_type in $mutation_types
    do
        file_prefix="../../asset/Evaluation_Datasets/${dataset}_${mutation_type}/${dataset}_${mutation_type}"
        echo ${file_prefix}

        CUDA_VISIBLE_DEVICES=${cuda_id} python run_classifier_WSD_sent.py \
        --task_name WSD \
        --eval_data_dir ${file_prefix}_test_sent_cls_ws.csv \
        --output_dir results \
        --bert_model results/Sent_CLS_WS \
        --do_test \
        --do_lower_case \
        --max_seq_length 512 \
        --train_batch_size 64 \
        --eval_batch_size 64 \
        --learning_rate 2e-5 \
        --num_train_epochs 6.0 \
        --seed 1314

        # convert output file as gold format
        python ../../src/wsd/glossbert/convert.py \
                --dataset ${dataset}_${mutation_type} \
                --input_file results/results.txt \
                --output_dir ${result_dir}  
    done
done

conda deactivate