import pandas as pd
import json
from src.prompts.LMmutate.config.chatglm_config import LLM


def filter_realistic(dataset, mut_type, en_preload_path, en_filter_path):
    # 读取包含 "sentence" 和 "mut_sentence" 列数据的CSV文件
    data = pd.read_csv(en_preload_path + '{0}_{1}_preload.csv'.format(dataset, mut_type))

    print("正在处理：{}".format(mut_type))

    # 定义check函数
    def check(row):
        mut_sentence = row['mut_sentence']

        check_prompt = '''
            You are a professional English assistant, your task is to check the given English sentences and give the results.
            The results include the following four types:
            • Realistic: Realistic sentences are grammatically and semantically correct.
            • Grammatical error: The sentence is obviously grammatically incorrect.
            • Counterintuitive: If a sentence clearly defies common sense, it is Counterintuitive.
            Sentence: {}
            Output format: {{"result":"Realistic" or" Grammatical error" or" Counterintuitive"}}
            Please output the result in json format.
            Output:
        '''.format(mut_sentence)
        output_string = LLM(check_prompt).replace("\n", "")

        # 查找 JSON 字符串的起始位置和结束位置
        start_index = output_string.find('{')
        end_index = output_string.rfind('}') + 1

        # 提取 JSON 字符串
        json_string = output_string[start_index:end_index]
        check_result = json.loads(json_string)

        result = check_result['result']

        return result

    count = 0  # 已处理的句子数
    dump_count = 0  # 去除的句子数

    for index, row in data.iterrows():
        count += 1
        if row["mut_tag"] == "MUTATED":
            try:
                if check(row) != 'Realistic':
                    # 'mut_type' 'word_index_list' 'mut_sentence' 'mut_tag' 'instance_count'
                    data.at[index, 'mut_type'] = ""
                    data.at[index, 'word_index_list'] = ""
                    data.at[index, 'mut_sentence'] = ""
                    data.at[index, 'mut_tag'] = "DUMP"
                    data.at[index, 'instance_count'] = ""
                    dump_count += 1
            except Exception as e:
                data.at[index, 'mut_type'] = ""
                data.at[index, 'word_index_list'] = ""
                data.at[index, 'mut_sentence'] = ""
                data.at[index, 'mut_tag'] = "DUMP"
                data.at[index, 'instance_count'] = ""
                dump_count += 1
                print("解析出错，争议句直接去除！")

        print("已处理{}个句子，去除{}个句子".format(count, dump_count))
    print("{}变异类型共筛除{}个句子".format(mut_type, dump_count))

    # 保存处理后的数据到新的CSV文件
    # data.to_csv('../../preload_data/filter_sentences/{0}_{1}_filter.csv'.format(dataset, mut_type), index=False)
    data.to_csv(en_filter_path+'{0}_{1}_filter.csv'.format(dataset, mut_type),
                index=False)


if __name__ == '__main__':
    # mutation_types = ["antonym", "comparative", "demonstrative", "number", "passivity", "that_this", "inversion",
    #                   "tenseplus", "modifier"]
    mutation_types = ["passivity", "that_this", "inversion", "tenseplus", "modifier"]

    en_preload_path = '../../preload_data/en_nocheck_preload/'
    en_filter_path = '../../preload_data/en_nocheck_filter_sentences/'
    for mut_type in mutation_types:
        filter_realistic("ALL", mut_type, en_preload_path, en_filter_path)
