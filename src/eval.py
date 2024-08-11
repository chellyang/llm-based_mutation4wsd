from utils.data_helper import mutation_types, get_key_format, datasets, wsd_systems
import pandas as pd
from utils.lm_helper import SEP
import numpy as np
from utils.data_helper import load_all_gold_predictions, load_all_predictions, load_mutated_sent_info, load_mutated_predictions
from bs4 import BeautifulSoup
import collections
import re


def post_process_helper(wsd_system, dataset, instance_word_dict, instance_pos_dict):
    key_format = get_key_format(wsd_system)

    golden_path = "asset/Evaluation_Datasets/{0}/{0}.gold.key.txt".format(dataset)
    ori_prediction_path = "result/predictions/{0}/{1}.results.key.txt".format(wsd_system, dataset)

    # 加载金标准数据和原始预测结果，并按照指定的格式进行处理和读取。
    gold_preds = load_all_gold_predictions(golden_path, key_format)
    ori_preds = load_all_predictions(ori_prediction_path, key_format)

    # 对不同的突变类型进行处理
    for mutation_type in mutation_types:
        print(mutation_type)
        # 首先根据突变类型加载对应的突变信息，然后构造了对应的预测结果路径，并使用 load_mutated_predictions 函数加载了突变后的预测结果
        mutated_sent_info = load_mutated_sent_info(dataset, mutation_type)
        mut_sent_ids = mutated_sent_info["id"].values

        mut_prediction_path = ori_prediction_path.replace(dataset, dataset+"_"+mutation_type, 1)
        mut_preds = load_mutated_predictions(mut_prediction_path, mut_sent_ids, key_format)
        # 创建了一个空字典 res 用于存储结果，并对处理后的数据进行了整合
        res = {
            "instance.id": [],
            "instance.word": [],
            "instance.pos": [],
            "sentence.id": [],
            "ori.prediction": [],
            "mut.prediction": [],
            "gold.prediction": []
        }
        for k, gold_prediction in gold_preds.items():
            sentence_id = k[:k.rfind(".")]
            if sentence_id not in mut_sent_ids:
                continue
            ori_prediction = ori_preds.get(k)
            mut_prediction = mut_preds.get(k)

            res["instance.id"].append(k)
            res["instance.word"].append(instance_word_dict[k])
            res["instance.pos"].append(instance_pos_dict[k])
            res["sentence.id"].append(sentence_id)
            res["ori.prediction"].append(ori_prediction)
            res["mut.prediction"].append(mut_prediction)
            res["gold.prediction"].append(SEP.join(list(gold_prediction)))

        df = pd.DataFrame(res)
        # 计算了一些列的值，包括预测错误的情况以及词义消歧 Bug 的情况
        df["pred"] = df.apply(lambda x: x["ori.prediction"] != x["mut.prediction"], axis=1)
        # regard as a true wsd bug if there is a bug in origianl or mutatant case
        df["broad.true"] = df.apply(lambda x: not ((x["ori.prediction"] is not None)
                              and (x["ori.prediction"] in x["gold.prediction"])
                              and (x["mut.prediction"] is not None)
                              and (x["mut.prediction"] in x["gold.prediction"])), axis=1)
        # regard as a true wsd bug only if there is bug in original case
        df["narrow.true"] = df.apply(lambda x: not ((x["ori.prediction"] is not None)
                              and (x["ori.prediction"] in x["gold.prediction"])), axis=1)
        # 保存处理后的结果到 CSV 文件
        pred_res_path = "result/predictions/{0}/{1}.results.key.csv".format(wsd_system, dataset+"_"+mutation_type)
        df.to_csv(pred_res_path, index=False)


def post_process():
    for dataset in datasets: 
        print("Dataset: ", dataset)
        instance_word_dict = collections.defaultdict(str)
        instance_pos_dict = collections.defaultdict(str)
        # 使用 BeautifulSoup 解析对应数据集的 XML 文件，并抽取出所有的 <sentence> 节点
        with open("asset/Evaluation_Datasets/{0}/{0}.data.xml".format(dataset)) as ef:
            soup = BeautifulSoup(ef, features="xml")
        nodes = soup.find_all("sentence")
        # 对每个 <sentence> 节点进行循环遍历，获取其中的 <instance> 节点，并将 <instance> 节点的 id 属性作为键，
        # 当前位置 cur_pos 和实例文本 child.string 作为值，分别添加到 instance_pos_dict 和 instance_word_dict 字典中
        for sentence_node in nodes:
            cur_pos = 0
            for child in sentence_node.children:
                if child.string == '\n':
                    continue
                if child.name == "instance":
                    instance_pos_dict[child["id"]] = cur_pos
                    instance_word_dict[child["id"]] = child.string
                cur_pos += 1
        # 对每个词义消歧系统调用 post_process_helper 方法进行进一步的处理
        for wsd_system in wsd_systems:
            print("Post Processing ", wsd_system)
            post_process_helper(wsd_system, dataset, instance_word_dict, instance_pos_dict)



# 对词义消歧结果进行后处理，包括加载数据、处理数据、计算预测结果正确与否等操作，
# 处理后的结果是一个包含处理后数据的 DataFrame，并且保存为 CSV 文件。
# 该结果将包括实例信息、原始预测结果、突变后预测结果、金标准的预测结果以及针对词义消歧 Bug 的判断情况等内容
if __name__ == '__main__':
    post_process()    