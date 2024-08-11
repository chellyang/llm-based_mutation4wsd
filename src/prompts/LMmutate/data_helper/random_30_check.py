import pandas as pd
import json
import numpy as np
from matplotlib.ticker import FuncFormatter

from src.prompts.LMmutate.config.chatglm_config import LLM
from src.prompts.LMmutate.data_helper.utils import get_tokens
import matplotlib.pyplot as plt
import glob
import seaborn as sns


def random_number(load_path, save_sample_path, number):
    # 读取已提取的mut_sentence数据和ALL.csv文件
    load_data = pd.read_csv(load_path)
    all_data = pd.read_csv('../../sentences/ALL.csv')

    # 根据mut_tag列的值筛选出mut_tag为"MUTATED"的行
    mutated_df = load_data[load_data['mut_tag'] == 'MUTATED']

    # 随机提取30个mut_sentence数据
    random_sample = mutated_df.sample(n=number)

    print(random_sample)
    # 提取对应的行号
    indices = random_sample.index.tolist()

    # 根据行号提取对应的sentence数据
    selected_data = all_data.loc[indices, 'sentence']

    # 将mut_sentence和对应的sentence数据拼接到一起
    merged_data = pd.concat([selected_data, random_sample['mut_sentence']], axis=1)

    # 保存拼接后的数据到新的CSV文件
    merged_data.to_csv(save_sample_path, index=False, header=['sentence', 'mut_sentence'])


def check_realistic(sample_path, mut_type):
    # 读取包含 "sentence" 和 "mut_sentence" 列数据的CSV文件
    data = pd.read_csv(sample_path)

    print("正在处理：{}".format(mut_type))

    # 定义check函数
    def check(row):
        # sentence = row['sentence']
        mut_sentence = row['mut_sentence']

        check_prompt = '''
            你是一个专业的英语助手，你的任务是对给定的英文句子进行语法和语义检查，给出检查结果并说明理由。
            检查结果包含下面三种：
            • Realistic: 句子在语法和语义上是正确的。
            • Grammatical error: 句子在语法上存在明显错误。
            • Counterintuitive: 句子违背了生活常识或逻辑矛盾。
            句子：{}
            输出格式：{{"result":"Realistic"或者"Grammatical error"或者"Counterintuitive","reason":理由(中文)}}
            请严格按json格式直接输出唯一结果。
            输出：   
        '''.format(mut_sentence)
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
    data[['check', 'reason']] = data.apply(lambda row: pd.Series(check(row)), axis=1)

    # 保存处理后的数据到原来的CSV文件
    data.to_csv(sample_path, index=False)


def check_standard(sample_path, mut_type):
    # 读取包含 "sentence" 和 "mut_sentence" 列数据的CSV文件
    data = pd.read_csv(sample_path)

    print("正在处理：{}".format(mut_type))
    mutation_types = {
        "antonym": "单词与其含义相反或相对的单词替换",
        "comparative": "将单词替换为其比较级形式或最高级形式",
        "demonstrative": "指示代词与具体名词之间的互换",
        "number": "数词和量词之间的替换",
        "passivity": "主动语态与被动语态之间的替换",
        "that_this": "this和that，these和those，it和they，them和us，you和me这些代词之间的互换",
        "inversion": "倒装变换",
        "tenseplus": "时态变换",
        "modifier": "添加修饰词"
    }

    # 定义check函数
    def check(row):
        sentence = row['sentence']
        mut_sentence = row['mut_sentence']

        check_prompt = '''
            你是一个专业的英语助手，你的任务是检查句子是否发生了指定的变换，给出结果并说明理由。
            检查结果包括：standard：句子发生了指定的变异；nonstandard：句子未发生指定的变异。
            原句：{0}
            变异句：{1}
            指定变换：{2}
            输出格式：{{"result":"standard"或者"nonstandard","reason":理由(中文)}}
            请严格按json格式直接输出唯一结果。
            输出：   
        '''.format(sentence, mut_sentence, mutation_types.get(mut_type))
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
    data[['standard', 'mut.reason']] = data.apply(lambda row: pd.Series(check(row)), axis=1)

    # 保存处理后的数据到原来的CSV文件
    data.to_csv(sample_path, index=False)


def percentage_calculate(sample_path, mut_type):
    # 读取CSV文件
    data = pd.read_csv(sample_path)

    # 统计check列中各个取值的频数
    check_counts = data['check'].value_counts()
    # 统计取值为Realistic的数量
    realistic_count = data[data['check'] == 'Realistic'].shape[0]

    # 计算各个取值的占比
    check_percentages = check_counts / check_counts.sum() * 100

    # 输出各个取值的频数和占比
    # print("各个取值的频数:")
    # print(check_counts)
    print("{}变异的错误占比:".format(mut_type))
    for value, percentage in check_percentages.items():
        print("{}: {:.2f}%".format(value, percentage))
    print("——————————————————————————")

    return realistic_count


def output_realistic_update(preload_list, filter_list):
    # 类别列表
    mutation_types = ["ant", "com", "dem", "num", "pas", "tha", "inv", "ten", "mod"]

    # 计算比率提升量
    improvement_rate = [(filter_count - preload_count) / preload_count * 100 for preload_count, filter_count in
                        zip(preload_list, filter_list)]
    # 创建柱状图
    fig, ax = plt.subplots()
    index = np.arange(len(mutation_types))
    bar_width = 0.35
    opacity = 0.8

    bar1 = plt.bar(index, preload_real_count_list, bar_width, alpha=opacity, color='g', label='Preload')
    bar2 = plt.bar(index, filter_real_count_list, bar_width, alpha=opacity, color='lightgreen', label='Filtered')

    # 在柱子上显示提升比率
    for i in range(len(mutation_types)):
        plt.text(x=index[i] - 0.3, y=filter_real_count_list[i] + 2, s=f'{improvement_rate[i]:.2f}%', color='k',
                 fontweight='bold')

    plt.xlabel('Mutation Types')
    plt.ylabel('Realistic Count')
    # plt.title('Realistic Count by Mutation Types')
    plt.xticks(index, mutation_types)
    plt.legend(loc='lower left')

    plt.tight_layout()
    plt.savefig('bar_realistic.pdf', format='pdf')
    plt.show()


def token_length_add(sample_path):
    # 读取CSV文件
    data = pd.read_csv(sample_path)
    # 计算每行的sentence的token数，并添加在该行的tokens.num列

    data['tokens.num'] = data['sentence'].apply(lambda x: len(get_tokens(x)))

    # 根据check列的值，添加新列值mut.error
    data['mut.error'] = data['check'].apply(lambda x: 0 if x == 'Realistic' else 1)

    # 保存处理后的数据到原来的CSV文件
    data.to_csv(sample_path, index=False)


def output_token_relation():
    # 读取所有包含 tokens.num 和 mut.error 列的 CSV 文件
    # all_files = glob.glob("../../preload_data/random30/preload/*.csv")
    all_files = glob.glob('../../preload_data/random30/token_preload/*.csv')
    # 类别列表
    mutation_types = ["antonym", "comparative", "demonstrative", "number", "passivity", "that_this", "inversion",
                      "tenseplus", "modifier"]
    # 创建一个空的 DataFrame 来存储所有数据
    combined_data = pd.DataFrame()

    # 循环读取每个 CSV 文件，提取所需的列，并合并到新的 DataFrame 中
    for file in all_files:
        data = pd.read_csv(file)
        combined_data = pd.concat([combined_data, data])

    # 调整"mut.error" 列的数据类型为 category
    combined_data["mut.error"] = combined_data["mut.error"].astype('category')

    # 创建一个包含 1 个子图的图表
    fig, axs = plt.subplots(2, 5, figsize=(20, 10))

    # 绘制每个类型中 tokens.num 和 mut.error 列的箱线图
    for i, file in enumerate(all_files):
        ax = axs[i // 5, i % 5]
        data = pd.read_csv(file)
        data["mut.error"] = data["mut.error"].astype('category')
        sns.boxplot(x="mut.error", y="tokens.num", data=data, ax=ax)
        ax.set_title(mutation_types[i], fontsize=18)

        # 设置 x 轴和 y 轴刻度标签字体大小
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        # 设置 x 轴和 y 轴标签字体大小
        ax.set_xlabel("mut.error", fontsize=16)
        ax.set_ylabel("tokens.num", fontsize=16)

        ax.set_ylim(0, 80)

    # 绘制所有类型总和的箱线图
    sns.boxplot(x="mut.error", y="tokens.num", data=combined_data, ax=axs[-1, -1])
    axs[-1, -1].set_title("All Types", fontsize=18)

    # 设置 x 轴和 y 轴刻度标签字体大小
    axs[-1, -1].tick_params(axis='x', labelsize=16)
    axs[-1, -1].tick_params(axis='y', labelsize=16)
    # 设置 x 轴和 y 轴标签字体大小
    axs[-1, -1].set_xlabel("mut.error", fontsize=16)
    axs[-1, -1].set_ylabel("tokens.num", fontsize=16)

    # plt.suptitle('Boxplot between Token Number and Mutate Error', fontsize=16)
    plt.tight_layout()
    plt.savefig('box_token.pdf', format='pdf')
    plt.show()


def output_standard_rate(mutation_types):
    # 读取所有的 CSV 文件
    df_list = [pd.read_csv('../../preload_data/random30/filter/ALL_{}_filter_sample.csv'.format(type)) for type in
               mutation_types]
    # 简化的类型名
    show_types = ["ant", "com", "dem", "num", "pas", "tha", "inv", "ten", "mod"]

    # 初始化列表，用于存储每个类型中'nonstandard'取值的占比
    nonstandard_ratio_list = []

    # 计算每个类型中'nonstandard'取值的占比
    for type, df in zip(mutation_types, df_list):
        total_count = len(df)
        nonstandard_count = (df['standard'] == 'nonstandard').sum()
        nonstandard_ratio = nonstandard_count / total_count
        nonstandard_ratio_list.append((type, nonstandard_ratio))

    # 对占比从高到低进行排序
    nonstandard_ratio_list.sort(key=lambda x: x[1], reverse=True)

    # 提取排序后的结果
    sorted_types = [x[0] for x in nonstandard_ratio_list]
    sorted_ratio = [x[1] for x in nonstandard_ratio_list]

    # 绘制柱状图
    bars = plt.bar(sorted_types, sorted_ratio, color='lightblue', width=0.6)

    # 自定义 y 轴标签格式为百分比
    def to_percent(y, position):
        return '{:.0%}'.format(y)

    # 获取当前的轴对象
    ax = plt.gca()
    formatter = FuncFormatter(to_percent)
    ax.yaxis.set_major_formatter(formatter)
    # 设置 y 轴上限为 100%
    plt.ylim(0, 1)
    plt.xlabel('Mutation Type')
    plt.ylabel('Ratio of Nonstandard')
    # plt.title('Ratio of Nonstandard in Each Type (Sorted)')
    plt.xticks(range(0, 9), [type[:3] for type in sorted_types])

    # 在每个柱子上显示数值
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), '{:.0%}'.format(bar.get_height()), ha='center',
                 va='bottom')

    plt.savefig('bar_standard.pdf', format='pdf')
    plt.show()
    # # 初始化字典，用于存储'standard'和'nonstandard'的数量
    # count_dict = {'standard': [0] * 9, 'nonstandard': [0] * 9}
    #
    # # 统计每种类型中'standard'和'nonstandard'的数量
    # for i, df in enumerate(df_list):
    #     count_dict['standard'][i] = (df['standard'] == 'standard').sum()
    #     count_dict['nonstandard'][i] = (df['standard'] == 'nonstandard').sum()
    #
    # # 计算每种类型中'standard'和'nonstandard'取值的占比
    # total_count = [count_dict['standard'][i] + count_dict['nonstandard'][i] for i in range(9)]
    # standard_ratio = [count_dict['standard'][i] / total_count[i] for i in range(9)]
    # nonstandard_ratio = [count_dict['nonstandard'][i] / total_count[i] for i in range(9)]
    #
    # # 绘制柱状图
    # fig, ax = plt.subplots()
    # bar_width = 0.35
    # index = range(1, 1)
    #
    # p1 = ax.bar(index, standard_ratio, label='Standard')
    # p2 = ax.bar(index, nonstandard_ratio, bottom=standard_ratio, label='Nonstandard')
    #
    # ax.set_xlabel('Type')
    # ax.set_ylabel('Ratio')
    # ax.set_title('Ratio of Standard and Nonstandard in Each Type')
    # ax.set_xticks([i + bar_width / 2 for i in index])
    # ax.set_xticklabels([type for type in mutation_types])
    # ax.legend()
    #
    # plt.show()


if __name__ == '__main__':

    dataset = 'ALL'

    # mutation_types = ["antonym", "comparative", "demonstrative", "number", "passivity", "that_this", "inversion",
    #                   "tenseplus", "modifier"]
    mutation_types = ["tenseplus", "modifier"]
    preload_real_count_list = []
    filter_real_count_list = []

    for mut_type in mutation_types:
        preload_path = '../../preload_data/{0}_{1}_preload.csv'.format(dataset, mut_type)
        filter_path = '../../preload_data/filter_sentences/{0}_{1}_filter.csv'.format(dataset, mut_type)
        filter_sample_path = '../../preload_data/random30/filter/{0}_{1}_filter_sample.csv'.format(dataset, mut_type)
        preload_sample_path = '../../preload_data/random30/preload/{0}_{1}_preload_sample.csv'.format(dataset, mut_type)
        preload_token_sample_path = '../../preload_data/random30/token_preload/{0}_{1}_preload_token_sample.csv'.format(
            dataset, mut_type)

        # ----------preload_data/random30/-----------
        # 取样、检查错误，计算token
        # --------------- preload(100) --------------
        # random_number(preload_path, preload_sample_path, 100)
        # check_realistic(preload_sample_path, mut_type)
        # check_standard(preload_sample_path, mut_type)
        # preload_real_count = percentage_calculate(preload_sample_path, mut_type)
        # preload_real_count_list.append(preload_real_count)

        # ----------- token_preload(200) ------------
        # 对变异句子原始样本计算token数探讨句子复杂性是否与变异错误有关
        # random_number(preload_path, preload_token_sample_path, 200)
        # check_realistic(preload_token_sample_path, mut_type)
        # token_length_add(preload_token_sample_path)

        # --------------- filter(100) ---------------
        # random_number(filter_path, filter_sample_path, 100)
        # check_realistic(filter_sample_path, mut_type)
        # check_standard(filter_sample_path, mut_type)
        # filter_real_count = percentage_calculate(filter_sample_path, mut_type)
        # filter_real_count_list.append(filter_real_count)

        for i in range(7, 10):
            top_p_sample_path = '../../preload_data/top_p_10/{0}/{1}_{2}_test.csv'.format(i, dataset, mut_type)
            check_realistic(top_p_sample_path, mut_type)


    # 分析结果
    # output_token_relation()
    # output_standard_rate(mutation_types)
    # output_realistic_update(preload_real_count_list, filter_real_count_list)
