#!/bin/bash

conda activate bem

cd download/wsd-biencoders

mutation_types="antonym comparative demonstrative number passivity that_this inversion tenseplus modifier"

# create result directory
result_dir="../../result/predictions/bem"

if [ -d ${result_dir} ]
then
echo ${result_dir}
else
mkdir -p ${result_dir}
fi

# read argument
read -rp "Enter CUDA ID (eg, -1, 0, 1, 2): " cuda_id
read -rp "Enter dataset name (multiple datasets seperated by space, eg, ALL):" datasets

# modify original code, delete dataset choice restriction
cp biencoder.py biencoder_modified.py
sed -i '65d' biencoder_modified.py

# test script
for dataset in $datasets
do
    CUDA_VISIBLE_DEVICES=${cuda_id} python biencoder_modified.py --data-path ../../asset --ckpt wsd-biencoder --eval --split ${dataset}
    cp wsd-biencoder/${dataset}_predictions.txt ${result_dir}/${dataset}.results.key.txt
    for mutation_type in $mutation_types
    do
        file_prefix="${dataset}_${mutation_type}"
        echo ${file_prefix}

        CUDA_VISIBLE_DEVICES=${cuda_id} python biencoder_modified.py --data-path ../../asset --ckpt wsd-biencoder --eval --split ${file_prefix}
        
        # copy results to local folder
        cp wsd-biencoder/${file_prefix}_predictions.txt \
        ${result_dir}/${file_prefix}.results.key.txt
    done
done

conda deactivate
cd ../../