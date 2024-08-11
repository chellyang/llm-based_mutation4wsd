import os
import re

import spacy
import pandas as pd

from bs4 import BeautifulSoup
from xml.etree import ElementTree as ET
from LMmutate.data_helper.utils import nltk_parser, get_tokens
from LMmutate.data_helper.updatexml import formatting_xml

# 加载 SpaCy 的英文模型
nlp = spacy.load("en_core_web_sm")


# 筛选原始数据集的目标词，去除四类目标词：词性冲突目标词、短语目标词、重复目标词、标点符号目标词
# 目前将原instance节点转化为wf节点时，并没有去除id属性
def filter_target_word(data_xml_path, new_data_xml_path):
    # 解析XML文件
    tree = ET.parse(data_xml_path)
    root = tree.getroot()

    # 记录所有改变的instance节点的id
    remove_instance_ids = []
    pos_change_instance_ids = []
    # 对原始XML中的每个sentence节点进行处理
    for text in root.findall('.//text'):
        text_id = text.get('id')
        print("==============================处理text：", text_id, "==============================")
        # 处理每个sentence节点
        for sentence_node in text.findall('.//sentence'):
            sentence_id = sentence_node.get('id')
            print("-----")
            print("处理句子：", sentence_id)
            # print("处理csv行：", row)

            # 首次遍历句子子节点，处理三类目标词
            # 存储sentence的各个节点
            node_list_first = []
            instance_lemmas = set()  # 记录已遇到的instance节点的lemma
            for child in sentence_node:
                if child.tag == "instance":
                    instance_id = child.get("id")
                    instance_lemma = child.get("lemma")

                    # 1. 处理包含空格的instance节点
                    if " " in child.text:
                        print("处理instance节点{}（短语目标词）：".format(instance_id), child.text)
                        # 解析短语的词源和词性
                        phrase_pos_lemma = nltk_parser(child.text.split(" "))
                        for i, word in enumerate(child.text.split(" ")):
                            wf_node = ET.Element("wf")
                            wf_node.text = word
                            wf_node.set("lemma", phrase_pos_lemma[i][2])
                            wf_node.set("pos", phrase_pos_lemma[i][1])
                            node_list_first.append(wf_node)
                        remove_instance_ids.append(instance_id)
                    # 2. 处理相同词源的instance节点
                    elif instance_lemma in instance_lemmas:
                        print("处理instance节点{}（词源相同目标词）：".format(instance_id), child.text)
                        child.tag = "wf"
                        node_list_first.append(child)
                        remove_instance_ids.append(instance_id)
                    # 3. 处理符号，以及类似函数调用的目标词
                    elif child.text == '%' or re.match(r'^(\w+)\(.*\)$', child.text):
                        print("处理instance节点{}（符号目标词）：".format(instance_id), child.text)
                        child.tag = "wf"
                        node_list_first.append(child)
                        remove_instance_ids.append(instance_id)
                    else:
                        node_list_first.append(child)
                        instance_lemmas.add(instance_lemma)
                else:
                    # 原wf节点直接存储
                    node_list_first.append(child)
            # 更新句子文本节点
            sentence_node.clear()
            sentence_node.extend(node_list_first)

            # 提取sentence节点的文本内容，组成句子
            text_list = []
            for child in sentence_node:
                if child.tag == "wf" or child.tag == "instance":
                    text_list.append(child.text)
            # 这里不能先合成原句，对原句字符串做分词，再解析，而应直接使用提取的单词列表解析，
            # 因为分词可能导致与xml文件的元素构成不符
            tokens_pos_lemma = nltk_parser(text_list)

            # 使用spacy进行准确的词性解析
            doc = nlp(" ".join(text_list))

            # 第二次遍历句子子节点，处理第四类目标词，因为短语目标词已被拆开，可以与token序号对应
            # 存储sentence的各个节点
            node_list_second = []
            for index, child in enumerate(sentence_node):
                if child.tag == "instance":
                    instance_id = child.get("id")
                    # 3. 处理词性不正确的instance节点
                    # 若对应索引目标词词性不在四种之类，将其转换为wf节点
                    if tokens_pos_lemma[index][1] not in ["ADV", "NOUN", "VERB", "ADJ"]:
                        print("词性冲突：遍历词{}，解析词{}，词性{}".format(child.text, tokens_pos_lemma[index][0],
                                                                         tokens_pos_lemma[index][1]))
                        print("处理instance节点{}（词性冲突）：".format(instance_id), child.text)
                        child.tag = "wf"
                        node_list_second.append(child)
                        remove_instance_ids.append(instance_id)
                    # 因为不能确定spacy和nltk分割token的一致性，使用nltk的词性进行判断，再由spacy作词性的准确赋予
                    elif tokens_pos_lemma[index][1] != child.get("pos") and \
                            tokens_pos_lemma[index][1] in ["ADV", "NOUN", "VERB", "ADJ"]:
                        # # 词性修正依赖于目标含义集中的词义，若目标含义集中具体含义不变，则最好不做词性修正，保证数据的准确性
                        # # 若词性与解析词性不同，且解析词性属于四种之类，替换正确词性
                        # print("处理instance节点{}（词性修正，不去除目标词）：".format(instance_id), child.text)
                        # # 确保是同一个token，否则以nltk为准;
                        # # spacy有其他类型，限制在四种之类；
                        # # "PROPN", "AUX"词性视为NOUN和VERB，为贴合bem模型
                        # if doc[index].text == tokens_pos_lemma[index][0] and doc[index].pos_ in ["ADV",
                        #                                                                          "NOUN",
                        #                                                                          "VERB",
                        #                                                                          "ADJ", "PROPN", "AUX"]:
                        #     # nltk词性判断并不准确，不同则以spacy为准
                        #     print("词性：{}——>{}".format(child.get("pos"), doc[index].pos_))
                        #     if doc[index].pos_ == "PROPN":
                        #         child.set("pos", "NOUN")
                        #     elif doc[index].pos_ == "AUX":
                        #         child.set("pos", "VERB")
                        #     else:
                        #         child.set("pos", doc[index].pos_)
                        # else:
                        #     print("词性：{}——>{}".format(child.get("pos"), tokens_pos_lemma[index][1]))
                        #     child.set("pos", tokens_pos_lemma[index][1])

                        # 直接去除目标词
                        child.tag = "wf"
                        node_list_second.append(child)
                        pos_change_instance_ids.append(instance_id)
                    else:
                        node_list_second.append(child)
                else:
                    # 原wf节点直接存储
                    node_list_second.append(child)
            # 更新句子文本节点
            sentence_node.clear()
            sentence_node.extend(node_list_second)
            sentence_node.set('id', sentence_id)

    # 报告修改的目标词数
    print("共去除目标词数：", len(remove_instance_ids) + len(pos_change_instance_ids))
    # print("共修改目标词词性数：", len(pos_change_instance_ids))
    # 将修改后的XML树写入新的xml文件
    tree.write(new_data_xml_path, encoding='utf-8')
    # 格式化xml文件，（不格式化bem可能出错）
    formatting_xml(new_data_xml_path)
    return remove_instance_ids


# 由原始数据集ALL.data.xml文件创建变异数据集ALL.csv,用于模型变异
def create_sentence_dataset(dataset):
    tree = ET.parse("../../asset/Evaluation_Datasets/{0}/{0}.data.xml".format(dataset))
    root = tree.getroot()

    sentence_nodes = root.findall(".//sentence")
    instance_list = []
    sentence_id_list = []
    for sn in sentence_nodes:
        sentence_id_list.append(sn.attrib['id'])
        instance_nodes = sn.findall(".//instance")
        instances_per_s = []
        for i in instance_nodes:
            instance = i.text.strip()
            instances_per_s.append(instance)
        instance_list.append(instances_per_s)

    print(instance_list)

    # 遍历所有sentence节点，提取其中的文本，构成句子，获得句子列表
    sentence_list = []
    for sentence_node in sentence_nodes:
        word_list = []
        for child in sentence_node:
            if child.tag == "wf" or child.tag == "instance":
                word_list.append(child.text)
        sentence = " ".join(word_list)
        sentence_list.append(sentence)

    # 将句子列表写入 CSV 文件
    output_file = './sentences/{}.csv'.format(dataset)

    # 创建数据框并将句子列表存入其中
    df = pd.DataFrame({'sentence.id': sentence_id_list, 'sentence': sentence_list, 'instance': instance_list})

    # 将数据框保存为 CSV 文件
    df.to_csv(output_file, index=False, header=True)

    print("CSV 文件保存完成:", output_file)


if __name__ == '__main__':
    # 数据集预处理，筛选目标词
    data_xml_file = "../../asset/Evaluation_Datasets/ALL/ALL.data.xml"  # 替换为实际的XML文件路径
    new_data_xml_file = "../../asset/Evaluation_Datasets/ALL/ALL.data.xml"
    changed_ids = filter_target_word(data_xml_path=data_xml_file, new_data_xml_path=new_data_xml_file)
    # 打印改变的instance节点的id
    print("Changed instance IDs:", changed_ids)

    # 根据数据集生成句子集，用于变异
    create_sentence_dataset('ALL')
