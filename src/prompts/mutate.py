from LMmutate.mutations.mut_passivity import passivity
from LMmutate.mutations.mut_demonstrative import demonstrative
from LMmutate.mutations.mut_that_this import that_this
from LMmutate.mutations.mut_number import number
from LMmutate.mutations.mut_comparative import comparative
from LMmutate.mutations.mut_antonym import antonym
from LMmutate.mutations.mut_inversion import inversion
from LMmutate.mutations.mut_tenseplus import tenseplus
from LMmutate.mutations.mut_modifier import modifier
import csv
import pandas as pd

# prompt_func_dict
prompt_func_dict = {
    "passivity": passivity,
    "demonstrative": demonstrative,
    "that_this": that_this,
    "number": number,
    "inversion": inversion,
    "tenseplus": tenseplus,
    "modifier": modifier,
    # 以下两个算子对该数据集变异性能不佳
    "comparative": comparative,
    "antonym": antonym,
}


def mutate_sentence(dataset_path, output_file, mut_type):
    # 读取csv文件
    file_path = dataset_path  # 替换为实际的文件路径
    df = pd.read_csv(file_path)

    # 获取sentence列，并存入列表
    sentence_list = df['sentence'].tolist()
    instance_list = df['instance'].tolist()

    ## 若生成中断，继续上次的csv文件生成使用
    # df = pd.read_csv('./preload_data/{0}_{1}_preload.csv'.format(dataset, mut_type))
    # # 计算 DataFrame 的行数（不包括表头）
    # num_rows = len(df.index)
    # # 去除已处理的句子
    # sentence_list = sentence_list[num_rows:]
    # instance_list = instance_list[num_rows:]

    # 根据mut_type选择对应的prompt_func
    prompt_func = prompt_func_dict[mut_type]

    # 遍历sentence列表，并将处理结果存入CSV文件
    fieldnames = ['mut_type', 'word_index_list', 'mut_sentence', 'mut_tag', 'instance_count']
    batch_size = 10
    count = 0  # 处理句子数
    instance_count = 0  # 目标词数
    right_count = 0  # 成功数
    false_count = 0  # 未发生变异句子数
    error_count = 0  # 出错数
    result_list = []

    # 首次创建CSV文件时，写入CSV文件头部
    with open(output_file, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()  # 写入CSV文件头部
        # # a 表示追加模式，如果文件不存在则创建文件，如果文件已存在则在文件末尾追加内容
        # with open(output_file, mode='a', newline='', encoding='utf-8') as csv_file:
        #     writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        for sentence, instance in zip(sentence_list, instance_list):
            print("正在处理句子:", sentence)
            sentence = sentence.replace('”', '``')
            sentence = sentence.replace("''", "``")

            count += 1
            # 注意：从csv读取的目标词列表是字符串格式，需要转换为列表类型
            instance = eval(instance)
            try:
                result = prompt_func(sentence, instance)
            except Exception as e:
                false_count += 1
                error_count += 1
                print("解析出错:", e)
                result = ("", "", "", "DUMP", "")

            if result[0] != '':
                right_count += 1
                instance_count += result[4]
            else:
                false_count += 1
            result_list.append(result)

            if count % batch_size == 0:
                for item in result_list:
                    writer.writerow \
                        ({'mut_type': item[0], 'word_index_list': item[1], 'mut_sentence': item[2], 'mut_tag': item[3],
                          'instance_count': item[4]})
                result_list = []

            print("已处理{}个句子，成功{}个，未变异{}个，出错{}个，目标词{}个".format(count, right_count, false_count,
                                                                                  error_count, instance_count))
        # 处理剩余的不足一整批的句子
        if result_list:
            for item in result_list:
                writer.writerow \
                    ({'mut_type': item[0], 'word_index_list': item[1], 'mut_sentence': item[2], 'mut_tag': item[3],
                      'instance_count': item[4]})

    print("处理结果已存入CSV文件:", output_file)


def random_dataset(dataset_path, sample_dataset_path):
    # 读取csv文件
    file_path = dataset_path  # 替换为实际的文件路径
    df = pd.read_csv(file_path)

    # 抽取随机样本，固定抽取50个句子
    df = df.sample(n=50, replace=True).reset_index(drop=True)
    df.to_csv(sample_dataset_path, index=False)


if __name__ == '__main__':
    # passivity demonstrative that_this number inversion tenseplus modifier comparative antonym
    dataset_path = 'sentences/ALL.csv'
    sample_dataset_path = 'sentences/ALL_random.csv'
    top_p = 7

    mut_types = ["passivity", "demonstrative", "that_this", "number", "inversion", "tenseplus", "modifier", "comparative", "antonym"]
    # mut_types = ["comparative"]
    for mut_type in mut_types:
        main_output_file = './preload_data/en_nocheck_preload/ALL_{}_preload.csv'.format(mut_type)
        test_output_file = './preload_data/top_p_10/{0}/ALL_{1}_test.csv'.format(top_p, mut_type)
        # 仅执行一次，抽取50个句子用于变异测试
        # random_dataset(dataset_path, sample_dataset_path)
        # mutate_sentence(sample_dataset_path, test_output_file, mut_type)

        # 主变异函数执行
        mutate_sentence(dataset_path, main_output_file, mut_type)
