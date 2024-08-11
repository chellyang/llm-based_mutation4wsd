import csv
import os
import xml.etree.ElementTree as ET
import xml.dom.minidom
import spacy

from tqdm import tqdm
from .utils import nltk_parser
from .utils import get_tokens

# 加载 SpaCy 的英文模型
nlp = spacy.load("en_core_web_sm")


# 该函数在变异后对各变异类型生成新的ALL_xxx.data.xml文件
def generate_new_xml(csv_file_path, xml_file_path, new_xml_file_path):
    # 解析原始XML文件
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # 读取csv文件，并存储每行的信息
    with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        csv_data = list(csv_reader)

    process_row_count = 0
    # 对原始XML中的每个sentence节点进行处理
    for text in tqdm(root.findall('.//text')):
        text_id = text.get('id')
        # print("==============================处理text：", text_id, "==============================")

        csv_data_remain = csv_data[process_row_count:]
        for sentence, row in zip(text.findall('.//sentence'), csv_data_remain):
            # print("-----")
            # print("处理句子：", sentence.get('id'))
            # print("处理csv行：", row)

            sentence_id = sentence.get('id')
            if row['mut_tag'] == "DUMP":
                process_row_count += 1
                # print("保留源节点")
                continue

            if row['mut_tag'] == "MUTATED":
                process_row_count += 1
                # print("创建新节点")
                # 根据mut_sentence创建新的sentence节点
                new_sentence = create_sentence_node_from_text(row['mut_sentence'], sentence)
                sentence.clear()  # 清空senctence的内容，以便插入新内容
                sentence.extend(new_sentence)  # 插入新的sentence内容
                sentence.set('id', sentence_id)

    # 将修改后的XML树写入新的xml文件
    tree.write(new_xml_file_path, encoding='utf-8')
    # 格式化xml文件，（不格式化bem可能出错）
    formatting_xml(new_xml_file_path)


# 对变异句创建新sentence节点
def create_sentence_node_from_text(mut_sentence, original_sentence):
    # original_sentence为xml句子节点,需要提取句子文本
    text_list = []
    for child in original_sentence:
        if child.tag == "wf" or child.tag == "instance":
            text_list.append(child.text)
    # 1这里不能对原句字符串做分词，再解析，而应直接使用提取的单词列表解析，分词可能导致与xml文件的元素构成不符
    # 2原句和变异句的同一token解析出的词源有差池，
    # 因此避免将句子的token列表用于解析，而只对单个token构成的列表进行解析,
    # 即每个单词独立解析，不包含上下文，以追求准确稳定性；
    # 后面的目标词和变异句的解析相同，解析仅针对词源，词性需全句解析取得
    # tokens_pos_lemma = nltk_parser(text_list)
    tokens_lemma_dict = {}
    for token in text_list:
        # tokens_lemma_dict[token] = nlp(token)[0].lemma_
        # 词源解析不能对xx-xx类词语解析为xx_xx,需要更改"-"符号，仅针对目标词做特殊修改
        if '-' in token:
            l = nltk_parser([token])[0][2]
            if nltk_parser([token])[0][1] != '.':
                l = l.replace("-", "_")
            tokens_lemma_dict[token.lower()] = l
        else:
            tokens_lemma_dict[token.lower()] = nlp(token.lower())[0].lemma_
    # 获取所有instance文本用于判断,每个instance节点的文本和id并存入字典
    instance_texts = []
    instances_dict = {}
    for instance_node in original_sentence.findall(".//instance"):
        instance_id = instance_node.get("id")
        instance_text = instance_node.text.lower()
        instance_texts.append(instance_text)
        # 解析单个目标词的词源
        instances_dict[tokens_lemma_dict[instance_text]] = instance_id
    # print("instance_dict:", instances_dict)

    # 获得所有目标词的词源
    instance_lemma = []
    for instance_text in instance_texts:
        # instance_lemma.append(nlp(instance_text)[0].lemma_)
        if '-' in instance_text:
            l = nltk_parser([instance_text])[0][2]
            if nltk_parser([instance_text])[0][1] != '.':
                l = l.replace("-", "_")
            instance_lemma.append(l)
        else:
            instance_lemma.append(nlp(instance_text.lower())[0].lemma_)
    # 获得变异句子token对应的词源列表
    # 词源可以单独来看，但词性必须以整句来看
    tokens = get_tokens(mut_sentence)
    mut_tokens_lemma = []
    mutant_tokens_pos_lemma = nltk_parser(tokens)
    # 使用spacy进行准确的词性解析
    # doc = nlp(" ".join(tokens))
    for token in tokens:
        # mut_tokens_lemma.append(nlp(token)[0].lemma_)
        if '-' in token:
            l = nltk_parser([token])[0][2]
            if nltk_parser([token])[0][1] != '.':
                l = l.replace("-", "_")
            mut_tokens_lemma.append(l)
        else:
            mut_tokens_lemma.append(nlp(token.lower())[0].lemma_)
    # 存储已被处理的目标词
    added_instance = []
    # 创建新sentence节点
    new_sentence = ET.Element("sentence")
    # 记录单词索引
    word_index = 0

    # 遍历tokens, 创建wf或者instance节点
    for token, lemma in zip(tokens, mut_tokens_lemma):
        # print(token, lemma)
        # print(instances_dict)
        # 检测是否为instance并且未处理过，这里以词的词源来判断
        # 若instance的单词在变异句中有多个，只给一个单词作为目标词节点
        if lemma in instance_lemma and lemma not in added_instance:
            added_instance.append(lemma)
            instance_id = instances_dict[lemma]
            new_instance = ET.SubElement(new_sentence, 'instance', id=instance_id)
            new_instance.text = token
            new_instance.set('pos', mutant_tokens_pos_lemma[word_index][1])
            new_instance.set('lemma', lemma)
        else:
            new_wf = ET.SubElement(new_sentence, 'wf')
            new_wf.text = token
            new_wf.set('pos', mutant_tokens_pos_lemma[word_index][1])
            new_wf.set('lemma', lemma)

        word_index += 1

    return new_sentence


def formatting_xml(xml_file_path):
    # 读取 XML 文件
    with open(xml_file_path, 'r', encoding='utf-8') as file:
        xml_str = file.read()
    # 解析XML，将其格式化
    dom = xml.dom.minidom.parseString(xml_str)
    formatted_xml = dom.toprettyxml()

    # 删除空白行
    lines = formatted_xml.split('\n')
    trimmed_xml = '\n'.join(line.strip() for line in lines if line.strip())

    # 将格式化后的 XML 保存到原始 XML 文件
    with open(xml_file_path, 'w', encoding='utf-8') as file:
        file.write(trimmed_xml)


if __name__ == '__main__':
    # pass
    # csv_file_path = 'example.csv'
    # xml_file_path = 'example_new.xml'
    # new_xml_file_path = 'example_new.xml'
    #
    # generate_new_xml(csv_file_path, xml_file_path, new_xml_file_path
    # mutation_types = ["antonym", "comparative", "demonstrative", "number", "passivity", "that_this", "inversion",
    #                   "tenseplus", "modifier"]
    # for type in mutation_types:
    all_xml_file_path = "../../../../asset/Evaluation_Datasets/ALL/ALL.data.xml"
    # xml_file_path = "../../../../asset/Evaluation_Datasets/ALL_{0}/ALL_{0}.data.xml".format(type)
    formatting_xml(all_xml_file_path)
