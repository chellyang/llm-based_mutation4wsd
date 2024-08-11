import csv
import re

import nltk
import pandas as pd
import spacy


def copy_csv_file(testset, number):
    origin_file = '../../sentences/ALL.csv'
    copy_file = './sentences/{}.csv'.format(testset)

    column_name = 'sentence'
    rows_to_copy = number

    # 使用 pandas 库读取 CSV 文件
    df = pd.read_csv(origin_file)

    # 从原始数据框中选择特定的列，并将结果保存到新的数据框中
    column_df = df[column_name].head(rows_to_copy)

    # 将结果写入新的 CSV 文件
    column_df.to_csv(copy_file, index=False, header=True)

    print("CSV文件复制完成:", copy_file)


def convert_to_list_of_tuples(input_string):
    # 删掉字符串中不需要的字符
    input_string = input_string.replace("[", "").replace("]", "").replace(" ", "").replace("'", "")
    # 按照 "),(" 分割字符串以获得每个元组的字符串表示
    tuple_strings = input_string.split("),(")

    result_list = []
    # 遍历每个元组的字符串表示
    for tuple_str in tuple_strings:
        # 删除括号并按照逗号分割来获得元组中的元素
        tuple_elements = tuple_str.replace("(", "").replace(")", "").split(",")
        if len(tuple_elements) == 2:
            # 将第二个元素从字符串转换为整数
            tuple_converted = (tuple_elements[0], int(tuple_elements[1]))
            result_list.append(tuple_converted)

    return result_list


def get_tokens(sentence):
    # 使用 NLTK 库的 word_tokenize() 函数来分割句子
    tokens = nltk.word_tokenize(sentence)
    # 分的更细一些
    # tokenizer = nltk.tokenize.WordPunctTokenizer()
    # tokens = tokenizer.tokenize(sentence)
    return tokens


from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn


def nltk_parser(tokens):
    lemmatizer = WordNetLemmatizer()
    # choose universal tagset to match xml file
    tagged = nltk.pos_tag(tokens, tagset='universal')
    tokens_pos_lemma = []
    for token, pos in tagged:
        wntag = wordnet_pos_code(pos)
        if wntag is None:  # not supply pos in case of None
            lemma = lemmatizer.lemmatize(token.lower())
        else:
            lemma = lemmatizer.lemmatize(token.lower(), pos=wntag)
        tokens_pos_lemma.append((token, pos, lemma))
    return tokens_pos_lemma


def wordnet_pos_code(tag):
    if tag in ["NOUN"]:
        return wn.NOUN
    elif tag in ["VERB"]:
        return wn.VERB
    elif tag in ["ADJ"]:
        return wn.ADJ
    elif tag in ["ADV"]:
        return wn.ADV
    else:
        return None


if __name__ == '__main__':
    pass
    # copy_csv_file('one',1)
    # 测试示例
    # input_string = "[('explains', '1'), ('assessed', '14'), ('reached', '20')]"


    # nlp = spacy.load("en_core_web_sm")
    # sentence = "Ringers , she added , are `` filled with the solemn intoxication that comes of intricate ritual faultlessly performed . `` ``"
    # output = get_tokens(sentence)
    # print(output)
    # # print(nltk_parser(output))
    # nlp_lemma_list = []
    # nlp_pos_list = []
    # nltk_lamma_list = []
    # nltk_pos_list = []
    # for token in output:
    #     nlp_lemma_list.append(nlp(token)[0].lemma_)
    #     nlp_pos_list.append(nlp(token)[0].pos_)
    #     nltk_lamma_list.append(nltk_parser([token])[0][2])
    #     nltk_pos_list.append(nltk_parser([token])[0][1])
    # print(nlp_lemma_list)
    # print(nlp_pos_list)
    # print(nltk_lamma_list)
    # print(nltk_pos_list)
    pattern = r'^(\w+)\(.*\)$'  # 正则表达式模式，匹配函数名后面带有括号的格式
    strings = ['sin()', 'cos()', 'sin(x)', 'log()', 'sqrt(x)','ssss(c)']
    for string in strings:

        match = re.match(pattern, string)
        if match:
            print("找到",string)