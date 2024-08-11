import csv
import re
import string
import subprocess

from bs4 import BeautifulSoup
import pandas as pd
# 具体的突变执行函数
from mutation.mutate import mutate_plural, mutate_gender, mutate_negative, mutate_tense
from prompts.LMmutate.data_helper.utils import convert_to_list_of_tuples, get_tokens
from prompts.LMmutate.data_helper.updatexml import generate_new_xml
from prompts.LMmutate.data_helper.updategoldkey import generate_new_gold_key
import spacy
from tqdm import tqdm
from typing import Iterable
import copy
from utils.lm_helper import load_model, nltk_parser, SEP
from utils.data_helper import mutation_types
import os

# 该程序是对数据集进行变异操作的顶层框架，主要针对ALL和Semcor数据集
# 函数调用基础流程：
# mutate_nodes——>mutate_sentence_node——>mutate_func(该函数字典具体调用mutation/mutate.py中的不同类型的变异函数）
#             ——>__mutate_sentence_node_helper（该函数参考变异函数得到的句子，对原句子模仿变异，按所需对结果作出修正）


# 突变函数字典，用于选取突变函数，在mutate_sentence_node函数调用
mutate_func_dict = {
    "plural": mutate_plural,
    "gender": mutate_gender,
    "tense": mutate_tense,
    "negative": mutate_negative,
}

classic_mutation = ['plural', 'gender', 'tense', 'negative']

en_nlp = load_model()


# 辅助对句子节点进行突变操作,间接修改xml文件
def __mutate_sentence_node_helper(sentence_node, tok_mut_sentence, mutation_subtype):
    # 使用nltk解析器解析突变后的句子，得到单词的词性和词形信息
    # mutant_tokens_pos_lemma变量形式（词，词性，词源）：
    # mutant_tokens_pos_lemma = [
    #     ('I', 'PRP', 'I'),
    #     ('am', 'VBP', 'be'),
    #     ('having', 'VBG', 'have'),
    #     ('some', 'DT', 'some'),
    #     ('banana', 'NN', 'banana')
    # ]
    mutant_tokens_pos_lemma = nltk_parser(tok_mut_sentence.split(SEP))

    # 遍历句子节点的子节点
    for child in sentence_node.children:
        # 跳过换行符节点
        if child.string == '\n':
            continue

        # 如果当前节点的文本与突变后的第一个单词不相等，则发生了突变
        if child.string != mutant_tokens_pos_lemma[0][0]:  # mutation happened
            if mutation_subtype == "plural":
                # 处理复数变异
                if child["lemma"] in ["many", "these", "those", "some"]:  # 当前单词是复数形式的代词，实现替换操作
                    if child.name == "instance":  # 避免替换变异
                        return True
                    child.string, child["pos"], child["lemma"] = mutant_tokens_pos_lemma[0]
                    mutant_tokens_pos_lemma.pop(0)
                elif mutant_tokens_pos_lemma[0][2] in ["a", "an"]:  # 如果突变单词信息列表中下一个单词的词元信息为 “a” 或 “an”，表示需要插入单词。
                    new_child = copy.copy(child)
                    if child.name == "instance":
                        new_child.name = "wf"
                        del new_child["id"]
                    new_child.string, new_child["pos"], new_child["lemma"] = mutant_tokens_pos_lemma[0]
                    child.insert_before(new_child)
                    child.insert_before('\n')
                    mutant_tokens_pos_lemma.pop(0)
                else:  # 如果当前子节点不需要替换或插入，则直接将当前子节点的字符串内容替换为突变单词信息列表的下一个单词
                    child.string = mutant_tokens_pos_lemma[0][0]
                    mutant_tokens_pos_lemma.pop(0)
            # 其他突变子类型的处理类似，根据具体的突变类型进行不同的处理
            elif mutation_subtype == "singular":
                if child["lemma"] in ["an", "a"]:  # delete word
                    child.extract()
                else:
                    child.string = mutant_tokens_pos_lemma[0][0]
                    mutant_tokens_pos_lemma.pop(0)
            elif mutation_subtype in ["positive", "SimplePast", "SimplePresent"]:
                if mutant_tokens_pos_lemma[0][1] != "VERB" or mutant_tokens_pos_lemma[0][2] == "do":  # insert word
                    new_child = copy.copy(child)
                    if child.name == "instance":
                        new_child.name = "wf"
                        del new_child["id"]
                    new_child.string, new_child["pos"], new_child["lemma"] = mutant_tokens_pos_lemma[0]
                    child.insert_before(new_child)
                    child.insert_before('\n')
                    mutant_tokens_pos_lemma.pop(0)
                else:
                    child.string = mutant_tokens_pos_lemma[0][0]
                    mutant_tokens_pos_lemma.pop(0)
            elif mutation_subtype == "negative":
                if child["pos"] != "VERB" or child["lemma"] == "do":  # delete word
                    child.extract()
                else:
                    child.string = mutant_tokens_pos_lemma[0][0]
                    mutant_tokens_pos_lemma.pop(0)
            elif mutation_subtype in ["female", "male"]:
                child.string = mutant_tokens_pos_lemma[0][0]
                mutant_tokens_pos_lemma.pop(0)
        else:
            # 如果当前节点的文本与突变后的第一个单词相等，则直接跳过，继续处理下一个单词
            mutant_tokens_pos_lemma.pop(0)

        # 如果突变后的单词已经处理完毕，则跳出循环
        if len(mutant_tokens_pos_lemma) == 0:
            break

    # 返回False表示突变处理完成
    return False


def mutate_sentence_node(node, doc, mutation_type):
    # 使用 spaCy 对句子进行处理，获取词元、词性、命名实体、依赖关系和句子头等信息
    spacy_tokens = [token.text_with_ws for token in doc]
    spacy_poss = [token.pos_ for token in doc]
    spacy_entities = [entity for entity in doc.ents]
    spacy_deps = [token.dep_ for token in doc]
    spacy_heads = [token.head for token in doc]

    # 根据传入的突变类型选择相应的突变函数
    mutate_func = mutate_func_dict[mutation_type]
    # 调用突变函数对句子进行突变处理，获取突变子类型、突变的单词、突变后的句子、突变后句子的空格分隔版本、词性标记和突变后的实例等信息
    mut_subtype, mut_words, _, tok_mut_sentence, tag, _ = mutate_func(-1, spacy_tokens, spacy_poss, spacy_entities,
                                                                      spacy_deps, spacy_heads)
    mut_tag = tag.value  # 获取突变的词性标记
    mutated_instances = extract_instances(mut_words, node, mut_tag)  # 提取突变后的实例

    mut_sentence = tok_mut_sentence.replace(SEP, " ")  # 将突变后的句子中的分隔符替换为空格

    if mut_tag == "MUTATED":  # 如果突变词性标记为"MUTATED"，则表示对句子进行局部突变
        if mutation_type == "gender" and len(mutated_instances) != 0:  # 对于性别突变，避免对实例进行突变
            return "", "", "", "", "DUMP", ""
        has_conflict = __mutate_sentence_node_helper(node, tok_mut_sentence, mut_subtype)
        # 避免在复数突变中对 many/some/these/those 进行实例突变
        if has_conflict:
            return "", "", "", "", "DUMP", ""
    return mut_subtype, mut_words, mut_sentence, tok_mut_sentence, mut_tag, mutated_instances


def mutate_nodes(dataset, mutation_type):
    # 打开指定数据集的 XML 文件，并使用 BeautifulSoup 库解析该 XML 文件，获取所有的 "sentence" 节点。
    with open("asset/Evaluation_Datasets/{0}/{0}.data.xml".format(dataset)) as ef:
        soup = BeautifulSoup(ef, features="xml")
    nodes = soup.find_all("sentence")

    # 初始化结果字段的字典 res，包括节点ID、实例单词、原始句子等各字段的信息
    res = {
        "id": [],
        "instance.word": [],
        "ori.sentence": [],
        "tok.ori.sentence": [],
        "mut.sentence": [],
        "tok.mut.sentence": [],
        "mut.type": [],
        "mut.subtype": [],
        "mut.word": [],
        "mut.tag": [],
        "instance.count": [],
        "mut.instance": []
    }

    # 遍历所有节点，获取并填充原始句子、原始句子的空格分隔版本、节点ID、突变类型等字段的信息。
    sentences_count = len(nodes)
    res["ori.sentence"] = [node.text.replace("\n", " ").strip() for node in nodes]
    res["tok.ori.sentence"] = [SEP.join(child.string for child in node.children if child.string != "\n") for node in
                               nodes]
    res["id"] = [node["id"] for node in nodes]
    res["mut.type"] = [mutation_type] * sentences_count

    # 使用 spaCy 处理原始句子，然后根据原始句子和指定的突变类型，调用 mutate_sentence_node 函数对句子进行突变处理
    docs: Iterable[spacy.tokens.Doc] = en_nlp.pipe(res["tok.ori.sentence"])

    # *****传统调用算法处理*****
    if mutation_type in classic_mutation:
        for node, doc in tqdm(zip(nodes, docs), total=sentences_count):
            # 获取突变子类型、突变的单词、突变后的句子、突变后句子的空格分隔版本、突变的词性标记、以及突变后的实例等信息
            mut_subtype, mut_words, mut_sentence, tok_mut_sentence, mut_tag, mutated_instances = mutate_sentence_node(
                node,
                doc,
                mutation_type)

            c = len(node.find_all('instance'))
            # 将获取的突变信息填充到结果字典 res 中
            res["mut.word"].append(mut_words)
            res["mut.instance"].append(mutated_instances)
            res["mut.sentence"].append(mut_sentence)
            res["tok.mut.sentence"].append(tok_mut_sentence)
            res["mut.tag"].append(mut_tag)
            res["instance.count"].append(c)
            res["mut.subtype"].append(mut_subtype)


    # *****调用语言模型处理*****
    else:
        # 读取大模型变异预处理文件
        # preload文件路径修改
        with open('./src/prompts/preload_data/filter_sentences/{0}_{1}_filter.csv'.format(dataset, mutation_type), 'r',encoding='utf-8') as file:
            csv_reader = csv.reader(file)

            # 跳过 CSV 文件的标题行
            next(csv_reader)

            for row in csv_reader:
                mut_type, word_index_list, mut_sentence, mut_tag, _ = row

                # 将突变词索引字符串转换为元组列表
                mut_words = convert_to_list_of_tuples(word_index_list)

                # 使用分词器进行分词
                tokens = get_tokens(mut_sentence)

                res["mut.word"].append(mut_words)
                res["mut.sentence"].append(' '.join(tokens))
                res["tok.mut.sentence"].append(SEP.join(tokens))
                res["mut.tag"].append(mut_tag)
                res["mut.subtype"].append(mut_type)
        # 处理节点其他信息
        for node, doc in zip(nodes, docs):
            c = len(node.find_all('instance'))
            # mutated_instances = extract_instances(mut_words, node, mut_tag)
            mutated_instances = ''
            res["mut.instance"].append(mutated_instances)
            res["instance.count"].append(c)

    # 生成关于突变操作的报告
    __mutation_report(dataset, mutation_type, res, len(nodes))
    # 输出变异结果文件：xml和txt文件
    if mutation_type in classic_mutation:
        output_mutation_result(dataset, mutation_type, soup, res)
    else:
        output_mutation_result_llm(dataset, mutation_type, res)


def output_mutation_result_llm(dataset, mutation_type, res):
    # 创建ALL_xxx文件夹,写入文件时，必须已存在外围文件夹
    mutated_dir = "asset/Evaluation_Datasets/{0}_{1}/".format(dataset, mutation_type)
    if not os.path.exists(mutated_dir):
        os.makedirs(mutated_dir)  # 如果路径不存在，就创建路径

    # 构建文件路径
    dataset_dir = "asset/Evaluation_Datasets/{0}/".format(dataset)

    # 已有文件
    mutated_preload_csv_path = "src/prompts/preload_data/filter_sentences/{0}_{1}_filter.csv".format(dataset, mutation_type)
    standard_gold_key_path = dataset_dir + "{0}_unprocessed.gold.key.txt".format(dataset)

    # 数据集ALL待生成文件
    dataset_xml_path = dataset_dir + "{0}.data.xml".format(dataset)
    dataset_gold_key_path = dataset_dir + "{0}.gold.key.txt".format(dataset)
    # 各个变异类型ALL_xxx待生成变异结果文件
    mutated_csv_path = mutated_dir + "{0}_{1}.data.csv".format(dataset, mutation_type)
    mutated_xml_path = mutated_dir + "{0}_{1}.data.xml".format(dataset, mutation_type)
    mutated_gold_key_path = mutated_dir + "{0}_{1}.gold.key.txt".format(dataset, mutation_type)

    # --------------------变异csv文件处理--------------------------
    # 变异生成新的ALL_xxx.data.csv文件
    # 将处理结果res字典写入CSV文件
    df = pd.DataFrame.from_dict(res, orient='index').T
    df.to_csv(mutated_csv_path, index=True)

    # 大模型变异：mut.instance列额外处理
    # 读取CSV文件
    df = pd.read_csv(mutated_csv_path)
    # 判断mut.tag列的值是否等于“DUMP”，若等于，则将mut.instance列的值改为""
    df.loc[df['mut.tag'] == 'DUMP', 'mut.instance'] = ""
    # 保存修改后的DataFrame回CSV文件
    df.to_csv(mutated_csv_path, index=False)
    # -----------------------------------------------------------
    # ---------------------变异xml文件处理-------------------------
    # 大模型变异: 变异生成新的ALL_xxx.data.xml文件
    generate_new_xml(mutated_preload_csv_path, dataset_xml_path, mutated_xml_path)

    with open(mutated_xml_path,encoding='utf-8') as ef:
        soup = BeautifulSoup(ef, features="xml")

    # 将BeautifulSoup对象转换为字符串并写入XML文件
    with open(mutated_xml_path, 'w', encoding="utf-8") as mf:
        str1 = str(soup).replace("<wf lemma='\"' pos=\".\">\"</wf>",
                                 "<wf lemma=\"&quot;\" pos=\".\">&quot;</wf>")  # 替换特定格式的字符串
        str2 = str1.replace('\'', '&apos;')  # 替换单引号为转义字符
        mf.write(str2)

    print(mutated_xml_path)
    # 验证生成的XML文件格式是否正确
    verify_xml_file(mutated_xml_path)
    # -----------------------------------------------------------
    # -------------------gold.key.txt文件生成----------------------
    # 数据集预处理后，ALL.gold.key文件重新生成
    if not os.path.exists(dataset_gold_key_path):
        generate_new_gold_key(dataset_xml_path, standard_gold_key_path, dataset_gold_key_path)

    # 大模型变异: 各变异类型生成新的ALL_xxx.gold.key文件
    generate_new_gold_key(mutated_xml_path, standard_gold_key_path, mutated_gold_key_path)

    # -----------------------------------------------------------


def extract_instances(mut_words, node, mut_tag):
    if mut_tag != "MUTATED":
        return ""
    node_index = 0
    mut_words_index = 0
    mut_instances = []
    for child in node.children:
        if child.string == "\n":
            continue
        if node_index == mut_words[mut_words_index][1]:
            # print(mut_words[mut_words_index])
            if child.name == "instance":
                mut_instances.append(child["id"])
            mut_words_index += 1
            if mut_words_index >= len(mut_words):
                break
        node_index += 1
    return SEP.join(mut_instances)


def output_mutation_result(dataset, mutation_type, soup, res):
    # 构建突变后文件夹路径
    mutated_dir = "asset/Evaluation_Datasets/{0}_{1}/".format(dataset, mutation_type)
    if not os.path.exists(mutated_dir):
        os.makedirs(mutated_dir)  # 如果路径不存在，就创建路径

    # 构建文件路径
    mutated_csv_path = mutated_dir + "{0}_{1}.data.csv".format(dataset, mutation_type)
    mutated_xml_path = mutated_dir + "{0}_{1}.data.xml".format(dataset, mutation_type)
    original_gold_txt_path = "asset/Evaluation_Datasets/{0}/{0}.gold.key.txt".format(dataset)
    mutated_gold_txt_path = mutated_dir + "{0}_{1}.gold.key.txt".format(dataset, mutation_type)

    # 将处理结果写入CSV文件
    df = pd.DataFrame.from_dict(res, orient='index').T
    df.to_csv(mutated_csv_path, index=True)

    # 将BeautifulSoup对象转换为字符串并写入XML文件
    with open(mutated_xml_path, 'w', encoding="utf-8") as mf:
        str1 = str(soup).replace("<wf lemma='\"' pos=\".\">\"</wf>",
                                 "<wf lemma=\"&quot;\" pos=\".\">&quot;</wf>")  # 替换特定格式的字符串
        str2 = str1.replace('\'', '&apos;')  # 替换单引号为转义字符
        mf.write(str2)

    print(mutated_xml_path)
    # 验证生成的XML文件格式是否正确
    verify_xml_file(mutated_xml_path)

    # 复制original_gold_txt_path文件中的内容到mutated_gold_txt_path文件中
    with open(original_gold_txt_path, "r") as og, open(mutated_gold_txt_path, 'w', encoding='utf-8') as gf:
        gold_text = og.readlines()
        gf.writelines(gold_text)


def __mutation_report(dataset, mutation_type, res, total_case_count):
    # 统计突变的句子数量和突变的句子比例
    mut_case_count = res["mut.tag"].count("MUTATED")
    mut_case_ratio = round(mut_case_count / total_case_count * 100, 2)

    direct_mut_instance_count = 0
    indirect_mut_instance_count = 0
    total_instance_count = 0
    for tag, instance_count, mut_instance in zip(res["mut.tag"], res["instance.count"], res["mut.instance"]):
        total_instance_count += instance_count
        if tag == "MUTATED":
            direct = len(mut_instance.split(SEP))
            indirect = instance_count - direct
            direct_mut_instance_count += direct
            indirect_mut_instance_count += indirect

    # 计算直接突变实例和间接突变实例的比例
    direct_mut_instance_ratio = round(direct_mut_instance_count / total_instance_count * 100, 2)
    indirect_mut_instance_ratio = round(indirect_mut_instance_count / total_instance_count * 100, 2)

    # 打印报告表格的表头和数据
    print("|dataset|mutation_type|#mutated sentence|#directly-mutated instance|#indirectly-mutated instance|")
    mutated_sentence_stat = '{0} ({1}%)'.format(mut_case_count, mut_case_ratio)
    direct_mutated_instance_stat = '{0} ({1}%)'.format(direct_mut_instance_count, direct_mut_instance_ratio)
    indirect_mutated_instance_stat = '{0} ({1}%)'.format(indirect_mut_instance_count, indirect_mut_instance_ratio)
    print("|" + "|".join([dataset, mutation_type, mutated_sentence_stat, direct_mutated_instance_stat,
                          indirect_mutated_instance_stat]) + "|")


# TODO, add sense checker
def verify_xml_file(xml_file_path):
    print("Verifying mutated XML file...")
    sentences = []
    s = []
    with open(xml_file_path, 'r', encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if line == '</sentence>':
                sentences.append(s)
                s = []

            elif line.startswith('<instance') or line.startswith('<wf'):
                word = re.search('>(.+?)<', line).group(1)
                lemma = re.search('lemma="(.+?)"', line).group(1)
                pos = re.search('pos="(.+?)"', line).group(1)

                # clean up data
                word = re.sub('&apos;', '\'', word)
                lemma = re.sub('&apos;', '\'', lemma)

                sense_inst = -1
                if line.startswith('<instance'):
                    sense_inst = re.search('instance id="(.+?)"', line).group(1)
                s.append((word, lemma, pos, sense_inst))
    # ubuntu
    subprocess.run(["java", "-cp", "asset/Data_Validation", "ValidateXML",
                    xml_file_path, "asset/Data_Validation/schema.xsd"])
    # windows
    # subprocess.run(["java", "-cp", "asset\Data_Validation", "ValidateXML",
    #                 xml_file_path, "asset\Data_Validation\schema.xsd"], shell=True)


if __name__ == '__main__':
    # 根据提供的突变类型和数据集，对数据集中的节点进行相应的突变操作。
    # 这些突变操作可能包括修改节点值、关联性质，删除节点或添加新的节点等改变数据集结构的操作。
    for mutation_type in mutation_types:
        mutate_nodes("ALL", mutation_type)
        # mutate_nodes("Semcor", mutation_type)
        # mutate_nodes("one", mutation_type)
