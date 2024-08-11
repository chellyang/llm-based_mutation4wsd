#!/bin/bash

conda activate ewiser

# run from root folder
cd download/ewiser
mutation_types="antonym comparative demonstrative number passivity that_this inversion tenseplus modifier"
#mutation_types="plural gender tense negative"

# create result directory
# 创建一个result目录。如果目录已经存在，则会打印出结果目录的路径。如果目录不存在，则会创建该目录。
result_dir="../../result/predictions/ewiser"

if [ -d ${result_dir} ]
then
echo ${result_dir}
else
mkdir -p ${result_dir}
fi

# read argument
read -rp "Enter CUDA ID (eg, cpu, cuda:0): " cuda_id
read -rp "Enter dataset name (multiple datasets seperated by space, eg, ALL):" datasets


xml_paths=""
# test script

# 遍历数据集和变异类型，拼接路径
for dataset in $datasets
do
    xml_path="../../asset/Evaluation_Datasets/${dataset}/${dataset}.data.xml"
    echo ${xml_path}
    xml_paths=${xml_paths}" "${xml_path}
    for mutation_type in $mutation_types
    do
        xml_path="../../asset/Evaluation_Datasets/${dataset}_${mutation_type}/${dataset}_${mutation_type}.data.xml"
        xml_paths=${xml_paths}" "${xml_path}
        echo ${xml_path}
    done
done     

# 调用模型执行词义消歧任务
python bin/eval_wsd.py --checkpoints ewiser.semcor.pt \
--xmls ${xml_paths} \
--predictions ${result_dir} \
--device ${cuda_id} \
--read-by sentence

conda deactivate