import re

import pandas as pd


def get_info(preload_csv_path):
    # 读取 CSV 文件
    df = pd.read_csv(preload_csv_path)

    # 获取文件总行数
    num_rows = len(df)

    # 计算 mut_tag 列各种取值的数量
    mut_tag_counts = df['mut_tag'].value_counts()

    # 计算 mut_type 列各种取值的数量
    mut_type_counts = df['mut_type'].value_counts()

    # 计算 instance_count 列值的和
    instance_count_sum = df['instance_count'].sum()

    # 使用正则表达式匹配以"number"开头和结尾的部分
    match = re.search(r'(?<=ALL_).*?(?=_filter.csv)', preload_csv_path)

    if match:
        extracted_str = match.group(0)

    # 打印结果
    print("=============={}==============".format(extracted_str))
    # print("Number of rows:\n", num_rows)
    # print("----------------------------------")
    print("mut_tag counts:\n", mut_tag_counts)
    # print("----------------------------------")
    # print("mut_type counts:\n", mut_type_counts.head())
    # print("----------------------------------")
    print("Sum of instance_count:\n", int(instance_count_sum))
    print("{}剩余目标词数:".format(extracted_str), int(instance_count_sum))

def whole_instance_counts(csv_path):
    # 读取 CSV 文件
    df = pd.read_csv(csv_path)

    # 将 instance 列转换为列表类型
    df['instance'] = df['instance'].apply(eval)

    # 计算字符串数目的总和
    total_string_count = 0
    for instance_list in df['instance']:
        total_string_count += len(instance_list)

    # 打印结果
    print("Total instance counts:", total_string_count)


if __name__ == '__main__':
    csv_path = "../../sentences/ALL.csv"
    mut_types = ['antonym','comparative','demonstrative','inversion','modifier','number','passivity','tenseplus','that_this']
    mut_types = ['demonstrative','inversion','modifier','number','passivity','tenseplus']
    # for type in mut_types:
    #     filter_csv_path = "ALL_{}_filter.csv".format(type)
    #     get_info(filter_csv_path)
    #     whole_instance_counts(csv_path)
    print(885+919+1032+876+1048+1036+822+996+909)
    print(3982 +5102+5320+5000+5517+4168+4764+5478+5120)
print(1154 + 1089 + 1140 + 1121 + 1132 + 1124 + 1098 + 1111 + 1111)

