
from typing import Any, Dict, Iterable, List, Literal, Tuple
import spacy
import pandas as pd
import tqdm
import os

from . import MutationResult
from .sentence import MutationSentence



MutationKind = Literal["gender", "plural", "tense", "negative"]
SEP = '@#*'

# 标记字符串都被去除首尾空格
def build_tokens(original_tokens: List[str]) -> List[str]:
    return [s.strip() for s in original_tokens]

from .plural import is_singular, is_plural
from .plural.mut import mutate_plural_to_singular, mutate_singular_to_plural
def mutate_plural(poly_word_index: int, original_tokens: List[str], original_pos: List[str],
                  original_entities: List[spacy.tokens.Span], original_deps, original_head) -> Dict[str, Any]:
    mutkind = []
    mutkind_word = []
    tokenized_mutant = original_tokens
    sentence = MutationSentence(tokenized_mutant)

    identical_align = [(i, i) for i in range(len(tokenized_mutant))]
    
    noun_indices = []

    for i in range(len(original_pos)):
        head = original_head[i].i
        if original_pos[i] == "NOUN" and original_deps[i] in ["nsubj", "dobj"] and head != i:
            if is_singular(tokenized_mutant, original_pos, original_entities, i) or \
                is_plural(tokenized_mutant, original_pos, original_entities, i):
                noun_indices.append(i)
    
    for noun_index in noun_indices:
        if noun_index == poly_word_index:
            continue
        # Found target NOUN
        if is_singular(tokenized_mutant, original_pos, original_entities, noun_index):
            mut_retcode = mutate_singular_to_plural(sentence, noun_index, original_pos, original_head, original_deps)
            if mut_retcode:
                mutkind.append("singular")
                mutkind_word.append((tokenized_mutant[noun_index], noun_index))

        elif is_plural(tokenized_mutant, original_pos, original_entities, noun_index):
            mut_retcode = mutate_plural_to_singular(sentence, noun_index, original_pos, original_head, original_deps)
            if mut_retcode:
                mutkind.append("plural")
                mutkind_word.append((tokenized_mutant[noun_index], noun_index))
            
    tokenized_mutant, align = sentence.get_result()

    align_map = {k:v for k, v in align}
    for i in noun_indices:
        mi = align_map[i]
        if mi != 0 and sentence[mi - 1].strip().lower() == ".":
            tokenized_mutant[mi] = tokenized_mutant[mi].capitalize()

    # Capitalize first char
    if tokenized_mutant[0][0].islower():
        tokenized_mutant[0] = tokenized_mutant[0].capitalize()

    # Output
    mutkind = list(set(mutkind))
    if len(mutkind) == 0:
        return "", mutkind_word, "".join(tokenized_mutant), SEP.join(build_tokens(tokenized_mutant)), MutationResult.DUMP, align
    elif len(mutkind) == 1:
        return mutkind[0], mutkind_word, "".join(tokenized_mutant), SEP.join(build_tokens(tokenized_mutant)), MutationResult.MUTATED, align
    else:
        return "", mutkind_word, "".join(tokenized_mutant), SEP.join(build_tokens(tokenized_mutant)), MutationResult.CONFLICT, align

from .gender.vocab import is_male_word, is_female_word
from .gender import has_person_name, replace_gender_token

def mutate_gender(poly_word_index: int, original_tokens: List[str], original_pos: List[str],
                  original_entities: List[spacy.tokens.Span], original_deps, original_head) -> Dict[str, Any]:
    gender = ""
    gender_words = list()
    mutant_tokens = list()
    pure_tokens = list()

    identical_align = [(i, i) for i in range(len(original_tokens))]

    if has_person_name(original_entities):
        return gender, gender_words, "".join(mutant_tokens), SEP.join(pure_tokens), MutationResult.NER, identical_align

    for (idx, (token, pos)) in enumerate(zip(original_tokens, original_pos)):
        current_gender = ""
        if pos in ["DET", "NOUN", "PRON", "ADJ"]:
            t = token.strip().lower()
            if is_male_word(t):
                current_gender = "male"
                gender_words.append((t, idx))
            elif is_female_word(t):
                current_gender = "female"
                gender_words.append((t, idx))
            else:
                mutant_tokens.append(token)
                continue

            # First occupation or consistent with previous gender words
            if gender == "" or (gender != "" and current_gender == gender):
                gender = current_gender
                mutant_tokens.append(replace_gender_token(token, pos))
            # Kick out sentences which contain conflict gender
            # For example: He has a daughter
            else:
                gender = ""
                pure_tokens = [s.strip() for s in mutant_tokens]
                return gender, gender_words, "".join(mutant_tokens), SEP.join(pure_tokens), MutationResult.CONFLICT, identical_align
        else:
            mutant_tokens.append(token)
    if gender != "":
        tag = MutationResult.MUTATED
    else:
        tag = MutationResult.DUMP
    pure_tokens = [s.strip() for s in mutant_tokens]
    return gender, gender_words, "".join(mutant_tokens), SEP.join(pure_tokens), tag, identical_align


from .tense import TenseType
from .tense.detection import TenseDetector
from .tense.mut import mutate_simple_past_to_simple_present, mutate_simple_present_to_simple_past

detector = TenseDetector()

def mutate_tense(poly_word_index: int, original_tokens, original_pos, original_entities, original_deps, original_head) -> Dict[str, Any]:
    mutkind = []
    mutkind_word = []
    tokenized_mutant = original_tokens
    sentence = MutationSentence(original_tokens)

    verb_indices = []
    # replace
    for i, token in enumerate(original_tokens):
        if original_deps[i] in ["ROOT", "conj"] and original_pos[i] == "VERB":
            verb_indices.append(i)

    for verb_index in verb_indices:
        token = original_tokens[verb_index]
        cleaned_token = original_tokens[verb_index].strip().lower()

        try:
            t = detector.detect_tense_verb(original_tokens, original_pos, original_deps, original_head, verb_index)
        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                raise e
            else:
                continue

        if t == TenseType.SimplePast:
            mut_retcode = mutate_simple_past_to_simple_present(sentence, verb_index, original_pos, original_head, original_deps)
            if mut_retcode:
                mutkind.append(t.name)
                mutkind_word.append((token, verb_index))
            continue
        elif t == TenseType.SimplePresent:
            mut_retcode = mutate_simple_present_to_simple_past(sentence, verb_index, original_pos, original_head, original_deps)
            if mut_retcode:
                mutkind.append(t.name)
                mutkind_word.append((token, verb_index))
            continue
    
    tokenized_mutant, align = sentence.get_result()
    mutkind = list(set(mutkind))
    if len(mutkind) == 0:
        return "", mutkind_word, "".join(tokenized_mutant), SEP.join(build_tokens(tokenized_mutant)), MutationResult.DUMP, align
    elif len(mutkind) == 1:
        return mutkind[0], mutkind_word, "".join(tokenized_mutant), SEP.join(build_tokens(tokenized_mutant)), MutationResult.MUTATED, align
    else:
        return "", mutkind_word, "".join(tokenized_mutant), SEP.join(build_tokens(tokenized_mutant)), MutationResult.CONFLICT, align

from .negative.detection import is_negative
from .negative.mut import mutate_negative_to_positive, mutate_positive_to_negative

def mutate_negative(poly_word_index: int, original_tokens, original_pos, original_entities, original_deps, original_head) -> Dict[str, Any]:
    mutkind = []
    mutkind_word = []
    tokenized_mutant = original_tokens
    sentence = MutationSentence(original_tokens)

    verb_indices = []
    # replace
    for i, token in enumerate(original_tokens):
        if original_deps[i] in ["ROOT", "conj"] and original_pos[i] in ["VERB", "AUX"]:
            verb_indices.append(i)

    for verb_index in verb_indices:
        token = original_tokens[verb_index]
        cleaned_token = original_tokens[verb_index].strip().lower()

        neg = is_negative(original_tokens, original_pos, original_deps, original_head, verb_index)

        if not neg:
            cleaned_tokens = [tok.lower().strip() for tok in sentence]
            if "?" in cleaned_tokens or "what" in cleaned_tokens or \
                    "why" in cleaned_tokens or "how" in cleaned_tokens:
                continue
            mut_retcode = mutate_positive_to_negative(sentence, verb_index, original_pos, original_head, original_deps)
            if mut_retcode:
                mutkind.append("positive")
                mutkind_word.append((cleaned_token, verb_index))
        else:
            mut_retcode = mutate_negative_to_positive(sentence, verb_index, original_pos, original_head, original_deps)
            if mut_retcode:
                mutkind.append("negative")
                mutkind_word.append((cleaned_token, verb_index))

    tokenized_mutant, align = sentence.get_result()
    mutkind = list(set(mutkind))
    if len(mutkind) == 0:
        return "", mutkind_word, "".join(tokenized_mutant), SEP.join(build_tokens(tokenized_mutant)), MutationResult.DUMP, align
    elif len(mutkind) == 1:
        return mutkind[0], mutkind_word, "".join(tokenized_mutant), SEP.join(build_tokens(tokenized_mutant)), MutationResult.MUTATED, align
    else:
        return "", mutkind_word, "".join(tokenized_mutant), SEP.join(build_tokens(tokenized_mutant)), MutationResult.CONFLICT, align


# def mutate(sentence_info_path: str, mutation_type: MutationKind) -> None:
#     # 加载英文语言模型
#     en_nlp = spacy.load('en_core_web_sm')
#
#     # 从CSV文件中读取句子数据
#     df = pd.read_csv(sentence_info_path)
#     sentences = list(df['ori.src'].values)
#
#     # 创建保存突变数据的字典
#     out_mutation_data = {
#         "ori.src": [],
#         "tok.ori.src": [],
#         "mut.src": [],
#         "tok.mut.src": [],
#         "mutkind": [],
#         "mutkind_word": [],
#         "tag": [],
#         "mut_align": [],
#     }
#
#     # 使用Spacy进行语言处理，获取句子的Doc对象
#     src_line_nlps: Iterable[spacy.tokens.Doc] = en_nlp.pipe(sentences)
#
#     # 遍历每个句子
#     for i, src_line_nlp in enumerate(tqdm.tqdm(src_line_nlps, total=len(sentences))):
#         # 获取当前句子的多义词和其位置
#         poly_word, poly_word_index = df['poly_word'].iloc[i].split("-")
#         poly_word_index = int(poly_word_index)
#
#         # 获取句子中的原始单词、词性标记、实体、依存关系和头部信息
#         original_tokens = [token.text_with_ws for token in src_line_nlp]
#         original_pos = [token.pos_ for token in src_line_nlp]
#         original_entities = [entity for entity in src_line_nlp.ents]
#         original_deps = [token.dep_ for token in src_line_nlp]
#         original_head = [token.head for token in src_line_nlp]
#
#         # 根据突变类型调用相应的突变函数
#         if mutation_type == "plural":
#             mutkind, mutkind_word, mutant, tokenized_mutant, tag, align = mutate_plural(poly_word_index,
#                                                                                         original_tokens, original_pos,
#                                                                                         original_entities,
#                                                                                         original_deps, original_head)
#         elif mutation_type == "gender":
#             mutkind, mutkind_word, mutant, tokenized_mutant, tag, align = mutate_gender(poly_word_index,
#                                                                                         original_tokens, original_pos,
#                                                                                         original_entities,
#                                                                                         original_deps, original_head)
#         elif mutation_type == "tense":
#             mutkind, mutkind_word, mutant, tokenized_mutant, tag, align = mutate_tense(poly_word_index, original_tokens,
#                                                                                        original_pos, original_entities,
#                                                                                        original_deps, original_head)
#         elif mutation_type == "negative":
#             mutkind, mutkind_word, mutant, tokenized_mutant, tag, align = mutate_negative(poly_word_index,
#                                                                                           original_tokens, original_pos,
#                                                                                           original_entities,
#                                                                                           original_deps, original_head)
#         else:
#             raise Exception("Unknown mutation type")
#
#         # 将突变后的数据保存到字典中
#         out_mutation_data["ori.src"].append(src_line_nlp.text)
#         out_mutation_data["tok.ori.src"].append(" ".join(build_tokens(original_tokens)))
#         out_mutation_data["mut.src"].append(mutant)
#         out_mutation_data["tok.mut.src"].append(tokenized_mutant)
#         out_mutation_data["mutkind"].append(mutkind)
#         out_mutation_data["mutkind_word"].append(mutkind_word)
#         out_mutation_data["tag"].append(tag.value)
#         out_mutation_data["mut_align"].append(",".join([f"{tup[0]}-{tup[1]}" for tup in align]))
#
#     # 将其他相关信息保存到字典中
#     out_mutation_data["ref"] = df.ref
#     out_mutation_data["poly_word"] = df.poly_word
#     out_mutation_data["corpus"] = df.corpus
#
#     # 生成保存突变数据的文件路径
#     mutated_path = os.path.join(
#         os.path.dirname(sentence_info_path),
#         os.path.basename(sentence_info_path).rsplit('.', maxsplit=1)[0] + f'_{mutation_type}_mutated.csv'
#     )
#
#     # 将突变数据存储到CSV文件中
#     df = pd.DataFrame(out_mutation_data)
#     df.to_csv(mutated_path, index=False)

