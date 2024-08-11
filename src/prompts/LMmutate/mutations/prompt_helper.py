import json
from ..config.chatglm_config import LLM
from ..data_helper.utils import get_tokens, nltk_parser


# 测试用
# from src.prompts.LMmutate.config.chatglm_config import LLM
# from src.prompts.LMmutate.data_helper.utils import get_tokens, nltk_parser


# 对llm的变异结果进行后处理
def get_all_mut_info(sentence, instance, output_string):
    # 查找 JSON 字符串的起始位置和结束位置
    start_index = output_string.find('{')
    end_index = output_string.rfind('}') + 1

    # 提取 JSON 字符串
    json_string = output_string[start_index:end_index]
    result_type_and_sentence = json.loads(json_string)

    # -------------------------------------------------------
    # 提取json中的值
    mut_type = result_type_and_sentence['type']
    # 1、检查是否发生变异
    if mut_type == '':
        print('无法变异')
        mut_tag = "DUMP"
        return mut_type, '', '', mut_tag, ''
    else:
        mut_tag = "MUTATED"

    mut_sentence = result_type_and_sentence['mut_sentence']
    mut_sentence = mut_sentence.replace('”', '``')
    mut_sentence = mut_sentence.replace("''", "``")

    # 2、检查变异后句子是否丢失了目标词，或者目标词词性解析有误
    if not check_mut_sentence(mut_sentence, instance):
        return '', '', '', 'DUMP', ''

    # --------------------------------------------------------

    # 寻找发生突变的单词和其在原句的下标
    mut_words_list = ['']
    word_index_list = ''

    # mut_words_list = find_mut_words(sentence, mut_sentence, mut_type)
    # word_index_list = get_mut_word_index(sentence, mut_words_list)
    # # 3、检查突变单词在原句是否有重复，保证找到的突变单词在原句唯一
    # if not word_index_list:
    #     print("突变单词有重复或未找到，忽略该次变异！")
    #     return '', '', '', 'DUMP', ''


    # --------------------------------------------------------
    instance_remain_count = get_instance_count(mut_sentence, instance)

    print("mut_type:", mut_type, "\n", "mut_sentence:", mut_sentence, "\n", "mut_word:", mut_words_list, "\n",
          "word_index:", word_index_list, "\n", "tag:", mut_tag, "\n", "instance_count:", instance_remain_count)

    return mut_type, word_index_list, mut_sentence, mut_tag, instance_remain_count


# 用于变异句子后获取突变单词
def find_mut_words(sentence, mut_sentence, input_type):
    prompt_get_mut_word = r'''The following original sentence has been transformed into a variant sentence after some 
    conversion. Your task is to extract the converted word from the original sentence or the word most related to the 
    conversion mut_word. 
    Original sentence: {} 
    Mutation sentence: {}
    Transformations that occur: {}
    Note: You need to extract the words in the original sentence, do not choose the words that are not in the original sentence.
    Output in json format.
    Output format: {{"mut_word":"word"}}
    Output:
      '''.format(sentence, mut_sentence, input_type)

    output_string1 = LLM(prompt_get_mut_word).replace("\n", "")

    # 查找 JSON 字符串的起始位置和结束位置
    start_index = output_string1.find('{')
    end_index = output_string1.rfind('}') + 1

    # 提取 JSON 字符串
    json_string2 = output_string1[start_index:end_index]

    result_word = json.loads(json_string2)

    mut_word_string = result_word['mut_word']

    # 变异单词列表，list[str]
    mut_words_list = mut_word_string.split(',')

    return mut_words_list


# 用于寻找突变单词的下标
def get_mut_word_index(sentence, mut_words_list):
    # 遍历原句，获得对应单词的下标
    tokens = get_tokens(sentence)

    word_index_list = []

    for word in mut_words_list:
        # 对原句中存在多个变异单词的输出错误,
        # 若给出的突变单词在原句中有多个，则无法判断是哪一个
        have_find = False
        for index, token in enumerate(tokens):
            # 找到突变单词，并且单词是首次被找到
            if token == word:
                if have_find is False:
                    word_index_list.append((word, index))
                    have_find = True
                else:
                    return []

    return word_index_list


# 检查变异句子是否丢失了突变实例
# 考虑到xml文件是重新生成，目标词以单词词源判断
def check_mut_sentence(mut_sentence, instances):
    mut_tokens = get_tokens(mut_sentence)
    mut_parsed = nltk_parser(mut_tokens)

    # 获得目标词词源列表
    instances_lemma = []
    for instance_token in instances:
        instances_lemma.append(nltk_parser([instance_token])[0][2])

    # 获得变异句tokens词源列表
    mut_tokens_lemma = []
    for mut_token in mut_tokens:
        mut_tokens_lemma.append(nltk_parser([mut_token])[0][2])

    # for target_word_lemma in instances_lemma:
    #     if target_word_lemma not in mut_tokens_lemma:
    #         print("检查出错（丢失目标词）：", target_word_lemma)
    #         return False

    # 变异后句子中，目标词词性检查，词性必须属于四种之一，避免生成xml文件时词性解析错误
    for mut_token_lemma, mut_parsed_tuple in zip(mut_tokens_lemma, mut_parsed):
        if mut_token_lemma in instances_lemma:
            if mut_parsed_tuple[1] not in ["VERB", "NOUN", "ADJ", "ADV"]:
                print("检查出错（突变句子目标词词性出错）：", mut_parsed_tuple)
                return False

    return True


def get_instance_count(mut_sentence, instances):
    mut_tokens = get_tokens(mut_sentence)

    # 获得目标词词源列表
    instances_lemma = []
    for instance_token in instances:
        instances_lemma.append(nltk_parser([instance_token])[0][2])

    # 获得变异句tokens词源列表
    mut_tokens_lemma = []
    for mut_token in mut_tokens:
        mut_tokens_lemma.append(nltk_parser([mut_token])[0][2])

    lose_instance_count = 0
    for instance_lemma in instances_lemma:
        if instance_lemma not in mut_tokens_lemma:
            lose_instance_count += 1

    return len(instances) - lose_instance_count


if __name__ == '__main__':
    # mut_words_list = ['apple', 'banana', 'orange']  # 单词列表
    # sentence = "There isn't an apple, a banana and an orange"
    # word_index_list = get_mut_word_index(sentence, mut_words_list)
    # print(word_index_list)

    # 测试示例
    sentence = "When they found it last winter , Dr. Vogelstein was dubious that the search was over ."
    mut_sentence = "When it was found last winter, Dr. Vogelstein was dubious that the search over was."
    target_words = ['found', 'last', 'winter', 'dubious', 'search', 'over']

    if not check_mut_sentence(mut_sentence, target_words):
        print("error")
