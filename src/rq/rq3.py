import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import venn
from src.utils.data_helper import mutation_types, datasets, wsd_systems
from pandas import read_csv


# 不同变异算子所报告的词义消歧错误是否存在交集
def rq3():
    for dataset in datasets:
        print(dataset)
        print("|model|mutation_type|wsd_bug_count|")
        print("|---|---|---|")
        # 创建大图，并指定大小为 30x20 英寸
        fig = plt.figure(figsize=(30, 20))
        # 创建一个网格布局，拥有 2 行和 6 列的子图空间
        gs = gridspec.GridSpec(2, 6)
        # 创建第一个子图，占据第一行的前两列
        ax1 = fig.add_subplot(gs[0, :2])
        # 创建第二个子图，占据第一行的第 3、4 列
        ax2 = fig.add_subplot(gs[0, 2:4])
        # 创建第三个子图，占据第一行的第 5、6 列
        ax3 = fig.add_subplot(gs[0, 4:])
        # 创建第四个子图，占据第二行的第 2、3 列
        ax4 = fig.add_subplot(gs[1, 1:3])
        # 创建第五个子图，占据第二行的第 4、5 列
        ax5 = fig.add_subplot(gs[1, 3:5])
        ax_list = [ax1, ax2, ax3, ax4, ax5]

        model_names = [wsd_system.upper() for wsd_system in wsd_systems]

        for wsd_system_i, wsd_system in enumerate(wsd_systems):
            labels = {}
            for mutation_type in mutation_types:
                pred_res_path = "../../result/predictions/{0}/{1}.results.key.csv".format(wsd_system,
                                                                                          dataset + "_" + mutation_type)
                df = read_csv(pred_res_path)
                # wsd_bug数量计算
                report_wsd_bugs = list((df[df["pred"]]["instance.id"].values))

                print("|" + "|".join([wsd_system, mutation_type, str(len(report_wsd_bugs))]) + "|")
                labels[mutation_type] = report_wsd_bugs

            # 基于突变类型创建 Venn 图的标签
            labels = venn.get_labels([labels[t] for t in mutation_types], fill=['number'])

            # 对于每个 WSD 系统和突变类型，创建并绘制 Venn 图
            # 参数 wsd_system_i 需根据具体情况进行修改，表示当前 WSD 系统的索引
            ax = venn.venn4_ax(ax_list[wsd_system_i],  # 在指定的子图中显示 Venn 图
                               labels,  # 标签数据
                               names=mutation_types,  # 突变类型的名称
                               legend=(wsd_system_i == 5),  # 是否显示图例（仅当 wsd_system_i 为第三个系统时显示）
                               fontsize=17)  # 标题字体大小

            # 设置 Venn 图的标题，标题为当前 WSD 系统的名称
            ax.set_title(model_names[wsd_system_i], y=-0.01, fontdict={'fontsize': 25})
        plt.subplots_adjust(top=0.8, bottom=0, hspace=-0.1)
        plt.margins(0, 0)
        fig.savefig(f"result/{dataset}_venn_out.pdf", bbox_inches='tight')


if __name__ == "__main__":
    rq3()
