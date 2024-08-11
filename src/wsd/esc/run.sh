#!/bin/bash

conda activate esc

cd download/esc

mutation_types="antonym comparative demonstrative number passivity that_this inversion tenseplus modifier"
#mutation_types="gender plural tense negative"
result_dir="../../result/predictions/esc"

if [ -d ${result_dir} ]
then
echo ${result_dir}
else
mkdir -p ${result_dir}
fi

read -rp "Enter CUDA ID (eg, -1, 0, 1, 2): " cuda_id
read -rp "Enter dataset name (multiple datasets seperated by space, eg, ALL):" datasets



# test script
# 已存在旧变异类型的预测文件，只需改名复制到统一的结果文件夹即可，运行旧类型将预测命令注释
for dataset in $datasets
do
    PYTHONPATH=$(pwd) python esc/predict.py --ckpt escher_semcor_best.ckpt \
        --dataset-paths "../../asset/Evaluation_Datasets/${dataset}/${dataset}.data.xml" \
        --prediction-type probabilistic --device ${cuda_id}

    # copy results to local folder
    cp predictions/${dataset}_predictions.txt \
    ${result_dir}/${dataset}.results.key.txt

    for mutation_type in $mutation_types
    do
        file_prefix="../../asset/Evaluation_Datasets/${dataset}_${mutation_type}/${dataset}_${mutation_type}"
        echo ${file_prefix}

        PYTHONPATH=$(pwd) python esc/predict.py --ckpt escher_semcor_best.ckpt \
        --dataset-paths ${file_prefix}.data.xml --prediction-type probabilistic --device ${cuda_id}
        
        # copy results to local folder
        cp predictions/${dataset}_${mutation_type}_predictions.txt \
        ${result_dir}/${dataset}_${mutation_type}.results.key.txt
    done
done


conda deactivate
