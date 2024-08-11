# Reference: https://github.com/SapienzaNLP/ewiser
from nltk.corpus import wordnet
from nltk.corpus.reader import WordNetError
from pathlib import Path
from collections import defaultdict
import os
from pandas import read_csv
import tqdm

patching_data = {
    'ddc%1:06:01::': 'dideoxycytosine.n.01.DDC',
    'ddi%1:06:01::': 'dideoxyinosine.n.01.DDI',
    'earth%1:15:01::': 'earth.n.04.earth',
    'earth%1:17:02::': 'earth.n.01.earth',
    'moon%1:17:03::': 'moon.n.01.moon',
    'sun%1:17:02::': 'sun.n.01.Sun',
    'kb%1:23:01::': 'kilobyte.n.02.kB',
    'kb%1:23:03::': 'kilobyte.n.01.kB',
}
datasets = ["ALL"]
# mutation_types = ["gender","negative", "plural", "tense"]
mutation_types = ["antonym", "comparative", "demonstrative", "number", "passivity", "that_this", "inversion", "tenseplus", "modifier"]
# mutation_types = ["demonstrative", "number", "passivity", "that_this", "inversion", "tenseplus", "modifier"]
wsd_systems = ["bem", "esc", "ewiser",  "glossbert", "syntagrank"]
# wsd_systems = ["esc"]


def patched_lemma_from_key(key, wordnet=wordnet):
    try:
        lemma = wordnet.lemma_from_key(key)
    except WordNetError as e:
        if key in patching_data:
            lemma = wordnet.lemma(patching_data[key])
        elif '%3' in key:
            lemma = wordnet.lemma_from_key(key.replace('%3', '%5'))
        else:
            raise e
    return lemma


def make_offset(synset):
    return "wn:" + str(synset.offset()).zfill(8) + synset.pos()


def convert(prediction, key_format):
    if key_format == "sense_key":
        return prediction
    else:
        return sensekey_to_wnsynset(prediction)


def sensekey_to_wnsynset(prediction):
    if prediction.startswith('wn:'):
        return prediction
    else:
        try:
            o = make_offset(patched_lemma_from_key(prediction).synset())
        except Exception:
            o = None
        else:
            return o


def wnsynset_to_sensekey(lemma, prediction):
    if prediction.startswith('wn:'):
        synset = wordnet.synset_from_pos_and_offset(prediction[-1], int(prediction[3:-1]))
        sense_keys = [l.key() for l in synset.lemmas() if
                      lemma.lower().replace(' ', '_') == l.key().lower().split('%')[0]]
        if len(sense_keys) != 1:
            print(sense_keys)
        return sense_keys[0]


def load_all_predictions(prediction_path, key_format):
    predictions = {}
    if not os.path.exists(prediction_path):
        return predictions
    for line in tqdm.tqdm(Path(prediction_path).read_text().splitlines()):
        pieces = line.strip().split(' ')
        if not pieces or len(pieces) <= 1:
            continue

        trg, prediction = pieces[0], pieces[1]
        predictions[trg] = convert(prediction, key_format)
    return predictions


def load_all_gold_predictions(golden_path, key_format):
    # One instance may have several golden keys
    gold_answers = defaultdict(set)
    if not os.path.exists(golden_path):
        return gold_answers
    for line in tqdm.tqdm(Path(golden_path).read_text().splitlines()):
        pieces = line.strip().split(' ')
        if not pieces:
            continue
        trg, *gold = pieces
        for g in gold:
            gold_answers[trg].add(convert(g, key_format))
    return gold_answers


def load_mutated_predictions(prediction_path, mut_sentence_ids, key_format):
    predictions = {}
    if not os.path.exists(prediction_path):
        return predictions
    for line in tqdm.tqdm(Path(prediction_path).read_text().splitlines()):
        pieces = line.strip().split(' ')
        if not pieces or len(pieces) <= 1:
            continue

        trg, prediction = pieces[0], pieces[1]
        sentence_id = trg[:trg.rfind(".")]
        if sentence_id in mut_sentence_ids:
            predictions[trg] = convert(prediction, key_format)
    return predictions


def load_mutated_gold_predictions(golden_path, mut_sentence_ids, key_format):
    # One instance may have several golden keys
    gold_answers = defaultdict(set)
    if not os.path.exists(golden_path):
        return gold_answers
    for line in tqdm.tqdm(Path(golden_path).read_text().splitlines()):
        pieces = line.strip().split(' ')
        if not pieces:
            continue
        trg, *gold = pieces
        sentence_id = trg[:trg.rfind(".")]
        if sentence_id in mut_sentence_ids:
            for g in gold:
                gold_answers[trg].add(convert(g, key_format))
    return gold_answers


def load_mutated_sent_info(dataset, mutation_type):
    mut_info = read_csv("asset/Evaluation_Datasets/{0}/{0}.data.csv".format(dataset + "_" + mutation_type))
    return mut_info[mut_info["mut.tag"].eq("MUTATED")]


def get_key_format(wsd_system):
    if wsd_system in ["ewiser", "syntagrank"]:
        key_format = "wordnet"
    else:
        key_format = "sense_key"
    return key_format


def f1_helper(tp, fp, tn, fn):
    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        precision = 0.

    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        recall = 0.

    try:
        f1 = (2 * precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1 = 0.
    return round(f1, 4), round(precision, 4), round(recall, 4)


def check_non_mutated_sentence(ori_prediction_path, mutation_type, dataset, key_format):
    mut_prediction_path = ori_prediction_path.replace(dataset, dataset + "_" + mutation_type, 1)
    mut_info = read_csv("asset/Evaluation_Datasets/{0}/{0}.data.csv".format(dataset + "_" + mutation_type))
    mutated_df = mut_info[mut_info["mut.tag"].eq("MUTATED")]
    mut_sentence_ids = mutated_df["id"].values
    golden_path = "asset/Evaluation_Datasets/{0}/{0}.gold.key.txt".format(dataset)

    ori_predictions = load_all_predictions(ori_prediction_path, key_format)

    mut_predictions = load_all_predictions(mut_prediction_path, key_format)
    gold_predictions = load_all_gold_predictions(golden_path, key_format)

    non_mutated_trigger = 0

    for k, gold_prediction in gold_predictions.items():
        # Filter out instances in sentences not mutated
        sentence_id = k[:k.rfind(".")]
        ori_prediction = ori_predictions.get(k)
        mut_prediction = mut_predictions.get(k)
        if sentence_id not in mut_sentence_ids:
            if ori_prediction != mut_prediction:
                non_mutated_trigger += 1
            #     print(k)
            #     print(mut_info[mut_info["id"].eq(sentence_id)]["ori.sentence"].values)
            #     print(mut_info[mut_info["id"].eq(sentence_id)]["mut.sentence"].values)

    print("Count of sentences non mutated but triggered:", non_mutated_trigger)


if __name__ == '__main__':
    print(wnsynset_to_sensekey("claim", "wn:06730563n"))
