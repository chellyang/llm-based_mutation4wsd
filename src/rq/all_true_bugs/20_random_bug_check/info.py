import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

dataset = 'ALL'

mutation_types = ["antonym", "comparative", "demonstrative", "number", "passivity", "that_this", "inversion",
                  "tenseplus", "modifier"]
wsd_systems = ['bem', 'esc', 'ewiser', 'glossbert', 'syntagrank']

show_mutation_types = ["ant", "com", "dem", "num", "pas", "tha", "inv", "ten", "mod"]
show_wsd_systems = ['bem', 'esc', 'ewi', 'glo', 'syn']


def get_bug_truth_info():
    bug_truth_list = []
    for mut_type in mutation_types:
        one_type_list = []
        for wsd_sys in wsd_systems:
            sample_path = "./{1}/{0}_{1}_{2}_sample.csv".format(dataset, mut_type, wsd_sys)
            df = pd.read_csv(sample_path)
            value_counts = df['check'].value_counts().to_dict()
            print(value_counts)
            true_count = value_counts.get(False)
            print(mut_type, wsd_sys, true_count)
            if true_count is None:
                true_count = 0
            one_type_list.append(true_count)
        bug_truth_list.append(one_type_list)

    # 创建绘图区域和子图
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 6))

    # 绘制每个子图
    for i in range(2):
        for j in range(5):
            ax = axes[i, j]
            if i == 1 and j == 4:
                # 计算每个子列表的和，并除以100获取占比
                values = [sum(sublist) / 100 for sublist in bug_truth_list]
                ax.bar(range(len(values)), values, width=0.6)
                ax.set_title('Sum Percentage', fontsize=14)
                ax.set_xticks(range(len(show_mutation_types)))  # 设置 x 轴刻度位置
                ax.set_xticklabels(show_mutation_types, fontsize=12)  # 设置 x 轴刻度标签
                ax.tick_params(axis='x', rotation=45)  # 设置 x 轴标签倾斜度为 45 度
            else:
                # 计算每个子列表中的数值比例（除以20）
                values = [num / 20 for num in bug_truth_list[i * 5 + j]]
                ax.bar(range(len(values)), values, width=0.6)
                ax.set_title(mutation_types[i * 5 + j], fontsize=14)
                ax.set_xticks(range(len(show_wsd_systems)))  # 设置 x 轴刻度位置
                ax.set_xticklabels(show_wsd_systems, fontsize=12)  # 设置 x 轴刻度标签

            # 自定义 y 轴标签格式为百分比
            def to_percent(y, position):
                return '{:.0%}'.format(y)

            formatter = FuncFormatter(to_percent)
            ax.yaxis.set_major_formatter(formatter)

            ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)

    plt.suptitle('', fontsize=16)
    # 调整子图的间距和标题位置
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('bar_bug_truth.pdf',format='pdf')
    # 显示图形
    plt.show()


if __name__ == '__main__':
    get_bug_truth_info()
