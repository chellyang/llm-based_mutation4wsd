import pandas as pd
import json
from src.prompts.LMmutate.config.chatglm_config import LLM


def check_bug(sample_path):
    # 读取包含 "sentence" 和 "mut_sentence" 列数据的CSV文件
    data = pd.read_csv(sample_path)

    # 定义check函数
    def check(row):
        instance = row['instance.word']
        sentence = row['sentence']
        mut_sentence = row['mut.sentence']

        check_prompt = '''
            你是一个专业的英语助手，你的任务是判断原句和变异句中各自的目标词的具体含义，若两个句子目标词的含义有所不同，输出结果为True，否则为False)，并给出理由。
            目标词：{0}
            原句：{1}
            变异句：{2}
            输出格式：{{"result":"True"或者"False","reason":理由(中文)}}
            请严格按json格式直接输出唯一结果。
            输出：   
        '''.format(instance, sentence, mut_sentence)
        output_string = LLM(check_prompt).replace("\n", "")

        try:
            # 查找 JSON 字符串的起始位置和结束位置
            start_index = output_string.find('{')
            end_index = output_string.rfind('}') + 1

            # 提取 JSON 字符串
            json_string = output_string[start_index:end_index]
            check_result = json.loads(json_string)

            result = check_result['result']
            reason = check_result['reason']
        except:
            result = 'need check'
            reason = 'need check'

        return result, reason

    # 对数据框中的每一行应用处理函数，将处理后的结果作为新的列值添加在对应行后
    data[['check_llm', 'reason']] = data.apply(lambda row: pd.Series(check(row)), axis=1)

    # 保存处理后的数据到原来的CSV文件
    data.to_csv(sample_path, index=False)


if __name__ == '__main__':

    dataset = 'ALL'

    # mutation_types = ["antonym", "comparative", "demonstrative", "number", "passivity", "that_this", "inversion",
    #                   "tenseplus", "modifier"]
    wsd_systems = ['bem', 'esc', 'ewiser', 'glossbert', 'syntagrank']
    # wsd_systems = ['esc', 'ewiser', 'glossbert', 'syntagrank']
    mutation_types = ["antonym"]

    for mut_type in mutation_types:
        print("#####正在处理：{}#####".format(mut_type))
        for wsd_sys in wsd_systems:
            print("#####正在处理：{0}/{1}#####".format(mut_type,wsd_sys))
            sample_path = './{1}/{0}_{1}_{2}_sample.csv'.format(dataset, mut_type, wsd_sys)
            check_bug(sample_path)
