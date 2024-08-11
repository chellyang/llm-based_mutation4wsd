from bs4 import BeautifulSoup
from pandas import read_csv
from wsd.syntagrank import disambiguate_tokens
import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import tqdm
from utils.data_helper import  mutation_types


# 模型消歧的顶层框架，具体只执行了使用api的syntagrank模型的消歧，其他本地部署的四个模型则在对应脚本运行
# src/esc/run.sh
def run_escher():
    pass

# src/ewiser/run.sh
def run_ewiser():
    pass

# src/glossbert/run.sh
def run_glossbert():
    pass

# 具体过程没复现出来
def run_ares():
    pass

# src/bem/run.sh
def run_bem():
    pass

# 网页爬取
def run_syntagrank(dataset, mutation_type, result_dir, save_batch=1000):
    if mutation_type != "":
        prefix = dataset+"_"+mutation_type
    else:
        prefix = dataset
    eval_res_path = result_dir + "/{0}.results.key.txt".format(prefix)
    
    if mutation_type != "":
        dataset = dataset + "_" + mutation_type
        mut_info = read_csv("asset/Evaluation_Datasets/{0}/{0}.data.csv".format(dataset))   
        mut_sentence_ids = mut_info[mut_info["mut.tag"].eq("MUTATED")]["id"].values
    with open("asset/Evaluation_Datasets/{0}/{0}.data.xml".format(dataset)) as ef:
        soup = BeautifulSoup(ef, features="xml")
    nodes = soup.find_all("sentence")

    if os.path.exists(eval_res_path):
        eval_res = {}
        print("Initializing a prediction dict...")
        # eval_res = load_predictions(eval_res_path)
        # print("Partial predictions loaded; continuing collecting disambiguation results from last point of interruption.")
    else:
        eval_res = {}
        print("Initializing a prediction dict...")

    # set Retry config
    s = requests.Session()
    s.mount('http://', HTTPAdapter(max_retries=Retry(total=10, allowed_methods=frozenset(['GET', 'POST'])))) 

    sentence_nodes = nodes
    for index, sentence_node in tqdm.tqdm(enumerate(sentence_nodes), total=len(sentence_nodes)):
        if index % save_batch == 0: # save periodically
            eval_res_list = []
            for id, synset_pos_and_offset in eval_res.items():
                eval_res_list.append(id + ' ' + synset_pos_and_offset + '\n')
            with open(eval_res_path, 'w') as rf:
                rf.writelines(eval_res_list)  
        
        sentence_id = sentence_node["id"]
        if (mutation_type != "") and (sentence_id not in mut_sentence_ids): # skip non-mutated sentence
            continue
        instance_example = ""
        tokens = []
        target_words = {}
        for child in sentence_node:
            if child.string == "\n":
                continue
            token_dict = {"word": child.string, "lemma": child["lemma"],
                        "pos": child["pos"]}
            if child.name == "instance":
                token_dict["id"] = child["id"]
                token_dict["isTargetWord"] = True
                target_words[child["id"]] = token_dict
                instance_example = child["id"]
            tokens.append(token_dict)
        if len(instance_example) > 0:
            if not eval_res.get(instance_example): # skip instances which have been predicted
                # post request only if it has not been predicted
                disambiguated_tokens = disambiguate_tokens(tokens, session=s)
                disambiguated_tokens = sorted(disambiguated_tokens)
                for id, synset_pos_and_offset in disambiguated_tokens:
                    eval_res[id] = synset_pos_and_offset  
    eval_res_list = []
    for id, synset_pos_and_offset in eval_res.items():
        eval_res_list.append(id + ' ' + synset_pos_and_offset + '\n')
    with open(eval_res_path, 'w') as rf:
        rf.writelines(eval_res_list)                   


wsd_func_dict = {
    "ares": run_ares, 
    "bem": run_bem,
    "escher": run_escher,
    "ewiser": run_ewiser,
    "glossbert": run_glossbert,
    "syntagrank": run_syntagrank
}
wsd_systems = wsd_func_dict.keys()


def run_wsd(wsd_system, dataset, mutation_type, result_dir):
    result_dir = result_dir + "/" + wsd_system
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    wsd_func_dict[wsd_system](dataset, mutation_type, result_dir)


if __name__ == '__main__':
    result_dir = "result/predictions"
    # run_wsd("syntagrank", "Semcor", "", result_dir)
    run_wsd("syntagrank", "ALL", "", result_dir)
    for mutation_type in mutation_types:
        run_wsd("syntagrank", "ALL", mutation_type, result_dir)
        # run_wsd("syntagrank", "Semcor", mutation_type, result_dir)


