import sys
import os
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from pandas import read_csv
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib_venn import venn2, venn3
from fpdf import FPDF

# add path environ
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

datasets = ["ALL"]
old_mutation_types = ["gender", "negative", "plural", "tense"]
mutation_types = ["antonym", "comparative", "demonstrative", "number", "passivity", "that_this", "inversion",
                  "tenseplus", "modifier"]
# mutation_types = ["demonstrative", "number", "passivity", "inversion", "tenseplus", "modifier"]
# wsd_systems = ["bem", "esc", "ewiser",  "glossbert", "syntagrank"]
wsd_systems = ["bem", "esc", "ewiser", "glossbert", "syntagrank"]

bug_df_list = []


# 方法有效性
def wsd_bug_report_llm(wsd_system, dataset, standard, res):
    mutation_types_df_list = []
    wsd_bug_df = pd.DataFrame()
    for mutation_type in mutation_types:
        pred_res_path = "../../result/predictions/{0}/{1}.results.key.csv".format(wsd_system,
                                                                                  dataset + "_" + mutation_type)
        if not os.path.exists("./all_true_bugs/{0}".format(mutation_type)):
            os.makedirs("./all_true_bugs/{0}".format(mutation_type))
        random_20_path = "./all_true_bugs/{0}/ALL_{0}_{1}_bug.csv".format(mutation_type, wsd_system)

        df = read_csv(pred_res_path)

        # 计算一些数量结果

        # 1. 计算instance.word和mut.prediction各自非空值的数量
        word_non_empty_count = df['instance.word'].count()
        mut_prediction_non_empty_count = df['mut.prediction'].count()

        # 2. 在mut.prediction非空的情况下进行计数
        mut_prediction_not_empty_df = df[df['mut.prediction'].notnull()]

        pred_true_count = mut_prediction_not_empty_df['pred'].sum()
        broad_true_count = mut_prediction_not_empty_df['broad.true'].sum()
        both_true_count = (mut_prediction_not_empty_df['pred'] & mut_prediction_not_empty_df['broad.true']).sum()

        both_true_df = mut_prediction_not_empty_df[
            mut_prediction_not_empty_df['pred'] & mut_prediction_not_empty_df['broad.true']]

        # 返回值wsd_bug_df,mutation_types_df_list
        wsd_bug_df = wsd_bug_df.append(both_true_df, ignore_index=True)
        mutation_types_df_list.append(both_true_df[['instance.id', 'instance.word', 'sentence.id']])

        # 在数据整理过程中存储一下消岐结果，用于后续采样，然后人工检查
        both_true_df[['instance.id', 'instance.word', 'sentence.id']].to_csv(random_20_path)

        pred_false_broad_true_count = (
                (~mut_prediction_not_empty_df['pred']) & mut_prediction_not_empty_df['broad.true']).sum()
        pred_true_broad_false_count = (
                (mut_prediction_not_empty_df['pred']) & ~mut_prediction_not_empty_df['broad.true']).sum()

        # 3. 提取instance.word非空的pred和broad.true列，赋给新的列
        # 这意味着在变异后依然存在的目标词作为样本集，
        # 总目标词数7253——>筛选后6667——>变异后存在的目标词集合
        non_empty_word_indices = df['instance.word'].notnull()
        new_df = df.loc[non_empty_word_indices, ['pred', 'broad.true', 'narrow.true']]

        # 选择预测的标准值
        y_trues = new_df["broad.true"] if standard == "broad" else new_df["narrow.true"]

        f1 = round(f1_score(y_trues, new_df["pred"]) * 100, 2)
        precision = round(precision_score(y_trues, new_df["pred"]) * 100, 2)
        recall = round((both_true_count / broad_true_count) * 100, 2)
        # # 模型消岐结果包含的是发生变异的原句中的目标词情况，因此考察变异后剩余的目标词需要以变异preload集中info为准，
        # # 存在预测值的目标词少于变异后剩余的目标词数，因为模型消岐时对于一些目标词的处理存在问题，跳过了少量目标词
        # 打印计算结果
        # print("| model   mutation_type   Precision   Recall   F1 |")
        # print("|-------------------------------------------------|")
        # print("|" + " & ".join([wsd_system, mutation_type, str(precision), str(recall), str(f1)]) + "|")
        # print("存在预测值的目标词总数(mut.prediction):", mut_prediction_non_empty_count)
        # print("在存在预测值的样本中:")
        # print("存在[词义消岐错误]的目标词个数(broad.true):", broad_true_count)
        # print("预测为[词义消岐错误]的个数(pred):", pred_true_count)
        # print("预测正确数(成功预测):", both_true_count, "预测错误数(未预测出):", pred_false_broad_true_count, "伪报正确数(错误预测):",
        #       pred_true_broad_false_count)
        # print("|-------------------------------------------------|")

        res["WSD_System"].append(wsd_system)
        res["Mutation_Type"].append(mutation_type)
        res["Effective_Prediction"].append(mut_prediction_non_empty_count)
        res["WSD_Target_Word"].append(broad_true_count)
        res["WSD_Predict"].append(pred_true_count)
        res["Success_Predict"].append(both_true_count)
        res["Miss_Predict"].append(pred_false_broad_true_count)
        res["Wrong_Predict"].append(pred_true_broad_false_count)
        res["F1"].append(f1)
        res["Precision"].append(precision)
        res["Recall"].append(recall)

    return wsd_bug_df, mutation_types_df_list


def wsd_bug_report(wsd_system, dataset, standard, res):
    wsd_bug_df = pd.DataFrame()
    for mutation_type in old_mutation_types:
        pred_res_path = "../../result/predictions/{0}/{1}.results.key.csv".format(wsd_system,
                                                                                  dataset + "_" + mutation_type)
        df = read_csv(pred_res_path)

        # 计算一些数量结果

        # 1. 计算instance.word和mut.prediction各自非空值的数量
        word_non_empty_count = df['instance.word'].count()
        mut_prediction_non_empty_count = df['mut.prediction'].count()

        # 2. 在mut.prediction非空的情况下进行计数
        mut_prediction_not_empty_df = df[df['mut.prediction'].notnull()]

        pred_true_count = mut_prediction_not_empty_df['pred'].sum()
        broad_true_count = mut_prediction_not_empty_df['broad.true'].sum()
        both_true_count = (mut_prediction_not_empty_df['pred'] & mut_prediction_not_empty_df['broad.true']).sum()

        both_true_df = mut_prediction_not_empty_df[
            mut_prediction_not_empty_df['pred'] & mut_prediction_not_empty_df['broad.true']]
        wsd_bug_df = wsd_bug_df.append(both_true_df, ignore_index=True)
        pred_false_broad_true_count = (
                (~mut_prediction_not_empty_df['pred']) & mut_prediction_not_empty_df['broad.true']).sum()
        pred_true_broad_false_count = (
                (mut_prediction_not_empty_df['pred']) & ~mut_prediction_not_empty_df['broad.true']).sum()

        # 3. 提取instance.word非空的pred和broad.true列，赋给新的列
        # 这意味着在变异后依然存在的目标词作为样本集，
        # 总目标词数7253——>筛选后6667——>变异后存在的目标词集合
        non_empty_word_indices = df['instance.word'].notnull()
        new_df = df.loc[non_empty_word_indices, ['pred', 'broad.true', 'narrow.true']]

        # 选择预测的标准值
        y_trues = new_df["broad.true"] if standard == "broad" else new_df["narrow.true"]

        f1 = round(f1_score(y_trues, new_df["pred"]) * 100, 2)
        precision = round(precision_score(y_trues, new_df["pred"]) * 100, 2)
        recall = round(recall_score(y_trues, new_df["pred"]) * 100, 2)
        # # 模型消岐结果包含的是发生变异的原句中的目标词情况，因此考察变异后剩余的目标词需要以变异preload集中info为准，
        # # 存在预测值的目标词少于变异后剩余的目标词数，因为模型消岐时对于一些目标词的处理存在问题，跳过了少量目标词
        # 打印计算结果
        # print("| model   mutation_type   Precision   Recall   F1 |")
        # print("|-------------------------------------------------|")
        # print("|" + " & ".join([wsd_system, mutation_type, str(precision), str(recall), str(f1)]) + "|")
        # print("存在预测值的目标词总数(mut.prediction):", mut_prediction_non_empty_count)
        # print("在存在预测值的样本中:")
        # print("存在[词义消岐错误]的目标词个数(broad.true):", broad_true_count)
        # print("预测为[词义消岐错误]的个数(pred):", pred_true_count)
        # print("预测正确数(成功预测):", both_true_count, "预测错误数(未预测出):", pred_false_broad_true_count, "伪报正确数(错误预测):",
        #       pred_true_broad_false_count)
        # print("|-------------------------------------------------|")

        res["WSD_System"].append(wsd_system)
        res["Mutation_Type"].append(mutation_type)
        res["Effective_Prediction"].append(mut_prediction_non_empty_count)
        res["WSD_Target_Word"].append(broad_true_count)
        res["WSD_Predict"].append(pred_true_count)
        res["Success_Predict"].append(both_true_count)
        res["Miss_Predict"].append(pred_false_broad_true_count)
        res["Wrong_Predict"].append(pred_true_broad_false_count)
        res["F1"].append(f1)
        res["Precision"].append(precision)
        res["Recall"].append(recall)

    return wsd_bug_df


def output_all_info(df, output_file_path):
    # 生成PDF文件

    with PdfPages(output_file_path) as pdf:
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.axis('tight')
        ax.axis('off')
        the_table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center',
                             colColours=['lightblue'] * len(df.columns))
        pdf.savefig()
        plt.close()


def output_mean_info(df):
    output_file_path = 'rq1_mean_info.pdf'
    # 根据 Mutation_Type 和 WSD_System 分组，计算其他列的平均值
    grouped_data = df.groupby(['Mutation_Type']).mean()
    # 设置 Pandas 输出的最大行和列数量
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)

    # 将分组后的数据保留行索引，并重置索引为默认值
    grouped_with_index = grouped_data.reset_index()

    print(grouped_with_index)
    # 设置不同的分组颜色
    colColours = ['lightblue', 'lightgreen', 'lightyellow', 'lightblue', 'lightgreen', 'lightyellow', 'lightblue',
                  'lightgreen', 'lightyellow', 'lightblue', 'lightgreen', 'lightyellow']

    with PdfPages(output_file_path) as pdf:
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.axis('tight')
        ax.axis('off')
        # 颜色参数：colColours=colColours[:len(grouped_with_index.columns)]
        the_table = ax.table(cellText=grouped_with_index.values, colLabels=grouped_with_index.columns, loc='center',
                             cellLoc='center', colColours=colColours[:len(grouped_with_index.columns)])
        pdf.savefig()
        plt.close()


def output_recall_info(df):
    # 根据 Mutation_Type 和 WSD_System 分组，计算其他列的平均值
    grouped_data = df.groupby(['Mutation_Type']).mean()
    # 将分组后的数据保留行索引，并重置索引为默认值
    grouped_with_index = grouped_data[['WSD_Predict', 'WSD_Target_Word', 'Recall']].reset_index()
    print(grouped_with_index)

    # 创建 PDF 文件
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=10)  # 调整字体大小

    # 加入表头
    cell_width = 34
    pdf.cell(cell_width * 4, 10, "WSD Bug Recall", 0, 1, 'C')
    pdf.cell(cell_width, 10, "Mutation Type", 1, 0, 'C')
    pdf.cell(cell_width, 10, "WSD Target (Mean)", 1, 0, 'C')
    pdf.cell(cell_width, 10, "WSD Predict (Mean)", 1, 0, 'C')
    pdf.cell(cell_width, 10, "Excavation (Mean)", 1, 0, 'C')
    pdf.ln()

    # 加入数据
    for i, row in grouped_with_index.iterrows():
        pdf.cell(cell_width, 10, str(row['Mutation_Type']), 1, 0, 'C')
        pdf.cell(cell_width, 10, "{:.2f}".format(row['WSD_Target_Word']), 1, 0, 'C')
        pdf.cell(cell_width, 10, "{:.2f}".format(row['WSD_Predict']), 1, 0, 'C')
        pdf.cell(cell_width, 10, "{:.2f}".format(row['Recall']), 1, 0, 'C')
        pdf.ln()

    # 保存PDF文件
    output_file_path = 'wsd_bug_recall.pdf'
    pdf.output(output_file_path)


def output_bug_info(df):
    # 构建新的列表
    new_list = []

    for mutation_type in mutation_types:
        # 获得当前变异类型对应不同消岐系统的值
        subset = df[df.index % 9 == mutation_types.index(mutation_type)]
        # row代表表格一行的值，是变异类型对应不同消岐系统的值
        row = subset['WSD_Predict'].tolist()
        # 添加总和
        row.append(sum(row))
        new_list.append(row)

    # 计算每列的总和
    column_sums = [sum(x) for x in zip(*new_list)]

    # 创建 PDF 文件
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=10)  # 调整字体大小

    # 加入表头
    cell_width = 28
    pdf.cell(cell_width * 7, 10, "WSD Predict Summary", 0, 1, 'C')
    pdf.cell(cell_width, 10, "Type", 1, 0, 'C')
    for system in wsd_systems + ["Total"]:
        pdf.cell(cell_width, 10, system, 1, 0, 'C')
    pdf.ln()

    # 加入数据
    for i, (mutation_type, values) in enumerate(zip(mutation_types, new_list)):
        pdf.cell(cell_width, 10, mutation_type, 1, 0, 'C')
        for value in values:
            pdf.cell(cell_width, 10, str(value), 1, 0, 'C')
        pdf.ln()

    # 添加总和行
    pdf.cell(cell_width, 10, "Total", 1, 0, 'C')
    for total_value in column_sums:
        pdf.cell(cell_width, 10, str(total_value), 1, 0, 'C')
    pdf.ln()

    # 保存 PDF 文件
    pdf.output("wsd_predict_summary.pdf")

    print("PDF 文件已保存成功。")


def output_recall_bar(df):
    # 类别列表
    show_types = ["ant", "com", "dem", "num", "pas", "tha", "inv", "ten", "mod"]
    # 将 DataFrame 中的数据按照每9行分成5大组存入二维数组
    num_rows = 9
    num_groups = 5
    grouped_data = []

    for i in range(num_groups):
        group_data = df.iloc[i * num_rows:(i + 1) * num_rows]['Recall'].values
        grouped_data.append(group_data)

    # 计算前5组每列的平均值作为第6组
    avg_data = np.mean(grouped_data, axis=0)
    grouped_data.append(avg_data)

    # 创建一个包含6个子图的图表
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))

    # 循环绘制每个子图
    for i, ax in enumerate(axs.flat):
        if i < 5:
            type_data = sorted(grouped_data[i], reverse=True)
            types_sorted = [x for _, x in
                            sorted(zip(grouped_data[i], show_types), key=lambda pair: pair[0], reverse=True)]
            ax.bar(types_sorted, type_data, color='orange')  # 设置柱子颜色为橙色
            ax.set_title(wsd_systems[i], fontsize=14)
            ax.set_xticklabels(types_sorted, rotation=45, fontsize=12)
            ax.set_ylim(0, 40)  # 设置 y 轴高度为 30
            ax.set_ylabel('Excavation(%)', fontsize=12)  # 设置 y 轴标签为'Recall'
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
        else:
            avg_data = sorted(grouped_data[5], reverse=True)
            avg_types_sorted = [x for _, x in
                                sorted(zip(grouped_data[5], show_types), key=lambda pair: pair[0], reverse=True)]
            ax.bar(avg_types_sorted, avg_data, color='orange')  # 设置柱子颜色为橙色
            ax.set_title("Average Excavation", fontsize=14)
            ax.set_xticklabels(avg_types_sorted, rotation=45, fontsize=12)
            ax.set_ylim(0, 40)  # 设置 y 轴高度为 30
            ax.set_ylabel('Excavation(%)', fontsize=12)  # 设置 y 轴标签为'Recall'
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)

    # 设置大标题
    # plt.suptitle('Recall Ability of Each Mutation Type', fontsize=16)
    # 调整布局
    plt.tight_layout()

    plt.savefig('bar_recall.pdf', format='pdf')
    # 显示图表
    plt.show()


def output_venn(df1, df2):
    # 计算相同行的个数
    common_rows = pd.merge(df1, df2, on=['instance.id', 'instance.word'])
    num_common_rows = len(common_rows)
    print("相同行的个数：", num_common_rows)
    print("相同的行：")
    print(common_rows)

    # 画出韦恩图
    venn2(subsets=(len(df1) - num_common_rows, len(df2) - num_common_rows, num_common_rows),
          set_labels=('df1', 'df2'))
    plt.show()


def output_all_venn(df_list_llm, df_list):
    # 创建一个大小为(12, 8)的图像，并设置布局为2行3列
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))

    # 设置子图的标题
    titles = ["bem", "esc", "ewiser", "glossbert", "syntagrank", "all wsd systems"]

    # 绘制前5个子图
    for i, ax in enumerate(axes.flatten()[:5]):
        # 获取对应位置的两个Dataframe
        df1 = df_list_llm[i]
        df2 = df_list[i]

        # 获取instance.id和instance.word列的元素
        set1 = set(df1['instance.id'])
        set2 = set(df2['instance.id'])

        # 计算并绘制venn图
        subvenn = venn2([set1, set2], set_labels=('our mutations', 'baseline'), ax=ax)

        # 修改交集数字字体大小
        subvenn.get_label_by_id('1').set_fontsize(12)
        subvenn.get_label_by_id('01').set_fontsize(12)
        subvenn.get_label_by_id('11').set_fontsize(12)

        # 修改所有子集数字字体大小
        for text in subvenn.set_labels:
            text.set_fontsize(14)

        ax.set_title(titles[i], fontsize=16)

    # 绘制第6个子图
    ax = axes.flatten()[5]
    # 计算两个列表中所有df总和的venn图
    total_set1 = set()
    total_set2 = set()
    for df1, df2 in zip(df_list_llm, df_list):
        total_set1.update(set(df1['instance.id']))
        total_set2.update(set(df2['instance.id']))
    venn = venn2([total_set1, total_set2], set_labels=('our mutations', 'baseline'), ax=ax)
    # 修改交集数字字体大小
    venn.get_label_by_id('1').set_fontsize(12)
    venn.get_label_by_id('01').set_fontsize(12)
    venn.get_label_by_id('11').set_fontsize(12)

    # 修改所有子集数字字体大小
    for text in venn.set_labels:
        text.set_fontsize(14)

    ax.set_title(titles[5], fontsize=16)

    # 去除空白的子图
    for ax in axes.flatten():
        if not ax.title.get_text():
            ax.set_axis_off()

    # 设置大标题
    # plt.suptitle('Overlap between Our Mutations and Baseline (duplicate removal)', fontsize=16)
    # 调整子图之间的间距
    plt.tight_layout()

    plt.savefig('venn.pdf', format='pdf')
    # 显示图像
    plt.show()


def output_all_pie(df_list_llm):
    # 设置子图的标题
    titles = ["bem", "esc", "ewiser", "glossbert", "syntagrank", "all wsd systems"]

    show_types = ["ant", "com", "dem", "num", "pas", "tha", "inv", "ten", "mod"]

    bug_counts_list = []

    # 绘制前5个子图
    for i in range(0, 5):
        # 获取第i个子列表的所有Dataframe
        df_list = df_list_llm[i]
        # 计算每个变异类型的数量
        df_counts = [len(df['instance.id']) for df in df_list]
        bug_counts_list.append(df_counts)

    # 计算五个子列表的汇总数量
    mutation_types_df_list = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                              pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]
    for sub_list in df_list_llm:
        mutation_types_df_list[0] = mutation_types_df_list[0].append(sub_list[0])
        mutation_types_df_list[1] = mutation_types_df_list[1].append(sub_list[1])
        mutation_types_df_list[2] = mutation_types_df_list[2].append(sub_list[2])
        mutation_types_df_list[3] = mutation_types_df_list[3].append(sub_list[3])
        mutation_types_df_list[4] = mutation_types_df_list[4].append(sub_list[4])
        mutation_types_df_list[5] = mutation_types_df_list[5].append(sub_list[5])
        mutation_types_df_list[6] = mutation_types_df_list[6].append(sub_list[6])
        mutation_types_df_list[7] = mutation_types_df_list[7].append(sub_list[7])
        mutation_types_df_list[8] = mutation_types_df_list[8].append(sub_list[8])

    sys5_all_df = pd.DataFrame()
    for df in mutation_types_df_list:
        sys5_all_df = sys5_all_df.append(df, ignore_index=True)

    total_counts = [len(df['instance.id'].drop_duplicates()) for df in mutation_types_df_list]

    bug_counts_list.append(total_counts)

    print(bug_counts_list)
    # 定义颜色列表，用于绘制饼图时设置各部分的颜色
    colors = ['skyblue', 'orange', 'lightgreen', 'pink', 'yellow', 'purple', 'brown', 'grey', 'teal']

    # 创建一个2行3列的图表，每个子图是一个饼图
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))

    # 遍历二维列表，为每个子图绘制饼图
    for i in range(2):
        for j in range(3):
            data = bug_counts_list[i * 3 + j]
            pie = axs[i, j].pie(data, labels=show_types, colors=colors, autopct='%1.1f%%',
                                startangle=140, textprops={'fontsize': 12})

            # 调整labels字体大小
            for label in pie[1]:
                label.set_fontsize(12)

            axs[i, j].set_title(titles[i * 3 + j], fontsize=14)

    plt.tight_layout()
    # 保存为PDF格式的图片
    plt.savefig('pie_counts.pdf', format='pdf')
    plt.show()


def random_bug_check():
    sentence_df = pd.read_csv("../prompts/sentences/ALL.csv")

    for mut_type in mutation_types:
        for wsd_sys in wsd_systems:
            # 从消岐错误报告文件中按变异类型，消岐系统抽样20个，得到目标词和其所在句的id值
            bug_path = "./all_true_bugs/{0}/ALL_{0}_{1}_bug.csv".format(mut_type, wsd_sys)
            if not os.path.exists("./all_true_bugs/20_random_bug_check/{0}".format(mut_type)):
                os.makedirs("./all_true_bugs/20_random_bug_check/{0}".format(mut_type))
            random_path = "./all_true_bugs/20_random_bug_check/{0}/ALL_{0}_{1}_sample.csv".format(mut_type, wsd_sys)
            mut_sentence_path = "../prompts/preload_data/ALL_{0}_preload.csv".format(mut_type)

            bug_df = pd.read_csv(bug_path)
            mut_sentence_df = pd.read_csv(mut_sentence_path)

            try:
                random_bug_df = bug_df.sample(n=20)[["instance.id", "instance.word", "sentence.id"]]
            except:  # 不足20个的，则直接提取所有的报告案例
                random_bug_df = bug_df[["instance.id", "instance.word", "sentence.id"]]

            ## 根据id值，找到对应的原句，将原句添加到df中
            # 如果有多个匹配的行，只会提取第一个匹配的sentence值，但在此不会出现该种情况
            random_bug_df["sentence"] = random_bug_df["sentence.id"].apply(
                lambda x: sentence_df[sentence_df["sentence.id"] == x]["sentence"].values[0] if len(
                    sentence_df[sentence_df["sentence.id"] == x]) > 0 else None
            )
            random_bug_df["sentence.index"] = random_bug_df["sentence.id"].apply(
                lambda x: sentence_df[sentence_df["sentence.id"] == x].index.values[0] if len(
                    sentence_df[sentence_df["sentence.id"] == x]) > 0 else None
            )

            # 根据行索引提取对应的mut.sentence数据，将变异句添加到df中
            indices = random_bug_df["sentence.index"].tolist()
            mut_sentence_list = mut_sentence_df["mut_sentence"].tolist()
            select_mut_sentence_list = [mut_sentence_list[i] for i in indices]
            random_bug_df['mut.sentence'] = select_mut_sentence_list

            random_bug_df_simple = random_bug_df.drop(['instance.id', 'sentence.id', 'sentence.index'], axis=1)
            # 新增一个名为 'check' 的列，作为检查结果
            random_bug_df_simple['check'] = ""
            random_bug_df_simple.to_csv(random_path, index=False)

    return


def rq1():
    for dataset in datasets:
        res = {
            "WSD_System": [],
            "Mutation_Type": [],
            "Effective_Prediction": [],
            "WSD_Target_Word": [],
            "WSD_Predict": [],
            "Success_Predict": [],
            "Miss_Predict": [],
            "Wrong_Predict": [],
            "F1": [],
            "Precision": [],
            "Recall": []
        }
        print(dataset)

        standard = "broad"
        all_bug_df_llm = pd.DataFrame()
        all_bug_df = pd.DataFrame()
        five_systems_bug_df_list_llm = []
        five_systems_bug_df_list = []

        systems_types_df_list = []  # 二维列表，外层为5个元素，代表5个消岐系统；内层为9个元素，代表9种变异类型；元素包含真消岐错误的id和word

        for wsd_system in wsd_systems:
            #每个模型（消岐系统）的bug报告案例
            # res针对新旧变异类型的两次赋值
            wsd_bug_df_llm, mutation_types_df_list = wsd_bug_report_llm(wsd_system, dataset, standard, res)
            # wsd_bug_df = wsd_bug_report(wsd_system, dataset, standard, res)

            # 综合所有模型
            all_bug_df_llm = all_bug_df_llm.append(wsd_bug_df_llm, ignore_index=True)
            # all_bug_df = all_bug_df.append(wsd_bug_df, ignore_index=True)

            five_systems_bug_df_list_llm.append(wsd_bug_df_llm)
            # five_systems_bug_df_list.append(wsd_bug_df)

            systems_types_df_list.append(mutation_types_df_list)

        # 使用nunique()函数计算唯一值的总数
        print(all_bug_df_llm['instance.id'].nunique())

        ## Venn图绘制 ####################################
        # 总图绘制
        # 保留instance.id和instance.word列，并去除重复值
        # df_list = [five_systems_bug_df_list[i][['instance.id', 'instance.word']].drop_duplicates() for i in range(5)]
        # df_list_llm = [five_systems_bug_df_list_llm[i][['instance.id', 'instance.word']].drop_duplicates() for i in
        #                range(5)]
        # output_all_venn(df_list_llm, df_list)
        #################################################

        ## 饼图绘制 ####################################
        # output_all_pie(systems_types_df_list)
        #################################################

        # 注意res在上面的代码中有两次赋值，在对新旧变异类型进行各自处理时注意注释掉另一方的res赋值
        df = pd.DataFrame(res)

        output_recall_bar(df)
        ## 实验结果表格绘制 ################################
        # 新旧变异类型注意修改存储地址
        # output_file_path = 'all_info.pdf'
        # output_all_info(df, output_file_path)
        # output_mean_info(df)
        # output_bug_info(df)
        # output_recall_info(df)
        #################################################


if __name__ == '__main__':
    rq1()
    # random_bug_check()
