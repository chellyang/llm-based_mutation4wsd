import pandas as pd
import xml.etree.ElementTree as ET
import os

# Ref: GlossBert
def generate_auxiliary_cls_ws(gold_key_file_name, train_file_name, train_file_final_name):

    sense_data = pd.read_csv("./wordnet/index.sense.gloss",sep="\t",header=None).values
    # print(len(sense_data))
    # print(sense_data[1])


    d = dict()
    for i in range(len(sense_data)):
        s = sense_data[i][0]
        pos = s.find("%")
        try:
            d[s[:pos + 2]].append((sense_data[i][0],sense_data[i][-1]))
        except:
            d[s[:pos + 2]]=[(sense_data[i][0], sense_data[i][-1])]

    # print(len(d))
    # print(len(d["happy%3"]))
    # print(d["happy%3"])
    # print(len(d["happy%5"]))
    # print(d["happy%5"])
    # print(len(d["hard%3"]))
    # print(d["hard%3"])



    train_data = pd.read_csv(train_file_name,sep="\t",na_filter=False).values
    # print(len(train_data))
    # print(train_data[0])

    gold_keys=[]
    with open(gold_key_file_name,"r",encoding="utf-8") as f:
        s=f.readline().strip()
        while s:
            tmp = s.split()[1:]
            gold_keys.append(tmp)
            s=f.readline().strip()
    # print(len(gold_keys))
    # print(gold_keys[6])

    with open(train_file_final_name,"w",encoding="utf-8") as f:
        f.write('target_id\tlabel\tsentence\tgloss\tsense_key\n')
        for i in range(len(train_data)):
            assert train_data[i][-2]=="NOUN" or train_data[i][-2]=="VERB" or train_data[i][-2]=="ADJ" or train_data[i][-2]=="ADV"
            orig_sentence = train_data[i][0].split(' ')
            start_id = int(train_data[i][1])
            end_id = int(train_data[i][2])
            sentence = []
            for w in range(len(orig_sentence)):
                if w == start_id or w == end_id:
                    sentence.append('"')
                sentence.append(orig_sentence[w])
            if end_id == len(orig_sentence):
                sentence.append('"')
            sentence = ' '.join(sentence)
            orig_word = ' '.join(orig_sentence[start_id:end_id])
            
            for category in ["%1", "%2", "%3", "%4", "%5"]:
                word = train_data[i][-3]
                query = word+category
                try:
                    sents = d[query]
                    gold_key_exist = 0
                    for j in range(len(sents)):
                        if sents[j][0] in gold_keys[i]:
                            f.write(train_data[i][3]+"\t"+"1")
                            gold_key_exist = 1
                        else:
                            f.write(train_data[i][3]+"\t"+"0")
                        f.write("\t"+sentence+"\t"+orig_word+" : "+sents[j][1]+"\t"+sents[j][0]+"\n")
                    assert gold_key_exist == 1
                except:
                    pass


def generate_csv(file_name):
    tree = ET.ElementTree(file=file_name)
    root = tree.getroot()

    sentences = []
    poss = []
    targets = []
    targets_index_start = []
    targets_index_end = []
    lemmas = []

    for doc in root:
        for sent in doc:
            sentence = []
            pos = []
            target = []
            target_index_start = []
            target_index_end = []
            lemma = []
            for token in sent:
                assert token.tag == 'wf' or token.tag == 'instance'
                if token.tag == 'wf':
                    for i in token.text.split(' '):
                        sentence.append(i)
                        pos.append(token.attrib['pos'])
                        target.append('X')
                        lemma.append(token.attrib['lemma'])
                if token.tag == 'instance':
                    target_start = len(sentence)
                    for i in token.text.split(' '):
                        sentence.append(i)
                        pos.append(token.attrib['pos'])
                        target.append(token.attrib['id'])
                        lemma.append(token.attrib['lemma'])
                    target_end = len(sentence)
                    assert ' '.join(sentence[target_start:target_end]) == token.text
                    target_index_start.append(target_start)
                    target_index_end.append(target_end)
            sentences.append(sentence)
            poss.append(pos)
            targets.append(target)
            targets_index_start.append(target_index_start)
            targets_index_end.append(target_index_end)
            lemmas.append(lemma)

    
    gold_keys = []
    with open(file_name[:-len('.data.xml')] + '.gold.key.txt', "r", encoding="utf-8") as m:
        key = m.readline().strip().split()
        while key:
            gold_keys.append(key[1])
            key = m.readline().strip().split()


    output_file = file_name[:-len('.data.xml')] + '.csv'
    with open(output_file, "w", encoding="utf-8") as g:
        g.write('sentence\ttarget_index_start\ttarget_index_end\ttarget_id\ttarget_lemma\ttarget_pos\tsense_key\n')
        num = 0
        for i in range(len(sentences)):
            for j in range(len(targets_index_start[i])):
                sentence = ' '.join(sentences[i])
                target_start = targets_index_start[i][j]
                target_end = targets_index_end[i][j]
                target_id = targets[i][target_start]
                target_lemma = lemmas[i][target_start]
                target_pos = poss[i][target_start]
                sense_key = gold_keys[num]
                num += 1
                g.write('\t'.join((sentence, str(target_start), str(target_end), target_id, target_lemma, target_pos, sense_key)))
                g.write('\n')


def generate_auxiliary_cls(gold_key_file_name, train_file_name, train_file_final_name):

    sense_data = pd.read_csv("./wordnet/index.sense.gloss",sep="\t",header=None).values



    d = dict()
    for i in range(len(sense_data)):
        s = sense_data[i][0]
        pos = s.find("%")
        try:
            d[s[:pos + 2]].append((sense_data[i][0],sense_data[i][-1]))
        except:
            d[s[:pos + 2]]=[(sense_data[i][0], sense_data[i][-1])]




    train_data = pd.read_csv(train_file_name,sep="\t",na_filter=False).values

    gold_keys=[]
    with open(gold_key_file_name,"r",encoding="utf-8") as f:
        s=f.readline().strip()
        while s:
            tmp = s.split()[1:]
            gold_keys.append(tmp)
            s=f.readline().strip()

    with open(train_file_final_name,"w",encoding="utf-8") as f:
        f.write('target_id\tlabel\tsentence\tgloss\tsense_key\n')
        for i in range(len(train_data)):
            assert train_data[i][-2]=="NOUN" or train_data[i][-2]=="VERB" or train_data[i][-2]=="ADJ" or train_data[i][-2]=="ADV"
            orig_sentence = train_data[i][0].split(' ')
            start_id = int(train_data[i][1])
            end_id = int(train_data[i][2])
            sentence = []
            for w in range(len(orig_sentence)):
                if w == start_id or w == end_id:
                    sentence.append('"')
                sentence.append(orig_sentence[w])
            sentence = ' '.join(sentence)
            orig_word = ' '.join(orig_sentence[start_id:end_id])
            
            for category in ["%1", "%2", "%3", "%4", "%5"]:
                word = train_data[i][-3]
                query = word+category
                try:
                    sents = d[query]
                    gold_key_exist = 0
                    for j in range(len(sents)):
                        if sents[j][0] in gold_keys[i]:
                            f.write(train_data[i][3]+"\t"+"1")
                            gold_key_exist = 1
                        else:
                            f.write(train_data[i][3]+"\t"+"0")
                        f.write("\t"+train_data[i][0]+"\t"+sents[j][1]+"\t"+sents[j][0]+"\n")
                    assert gold_key_exist == 1
                except:
                    pass


def generate_auxiliary_token_cls(gold_key_file_name, train_file_name, train_file_final_name):

    sense_data = pd.read_csv("./wordnet/index.sense.gloss",sep="\t",header=None).values



    d = dict()
    for i in range(len(sense_data)):
        s = sense_data[i][0]
        pos = s.find("%")
        try:
            d[s[:pos + 2]].append((sense_data[i][0],sense_data[i][-1]))
        except:
            d[s[:pos + 2]]=[(sense_data[i][0], sense_data[i][-1])]





    train_data = pd.read_csv(train_file_name,sep="\t",na_filter=False).values


    gold_keys=[]
    with open(gold_key_file_name,"r",encoding="utf-8") as f:
        s=f.readline().strip()
        while s:
            tmp = s.split()[1:]
            gold_keys.append(tmp)
            s=f.readline().strip()


    with open(train_file_final_name,"w",encoding="utf-8") as f:
        f.write('target_id\tlabel\tsentence\tgloss\ttarget_index_start\ttarget_index_end\tsense_key\n')
        for i in range(len(train_data)):
            assert train_data[i][-2]=="NOUN" or train_data[i][-2]=="VERB" or train_data[i][-2]=="ADJ" or train_data[i][-2]=="ADV"

            for category in ["%1", "%2", "%3", "%4", "%5"]:
                word = train_data[i][-3]
                query = word+category
                try:
                    sents = d[query]
                    gold_key_exist = 0
                    for j in range(len(sents)):
                        if sents[j][0] in gold_keys[i]:
                            f.write(train_data[i][3]+"\t"+"1")
                            gold_key_exist = 1
                        else:
                            f.write(train_data[i][3]+"\t"+"0")
                        f.write("\t"+train_data[i][0]+"\t"+sents[j][1]+"\t"+str(train_data[i][1])+"\t"+str(train_data[i][2])+"\t"+sents[j][0]+"\n")
                    assert gold_key_exist == 1
                except:
                    pass


if __name__ == "__main__":
    # excute from root folder
    # datasets = ['ALL', 'Semcor']
    datasets = ['ALL']
    # mutation_types = ["", "plural", "gender", "tense", "negative"]
    # mutation_types = ["", "antonym", "comparative", "demonstrative", "number", "passivity", "that_this", "inversion",
    #                   "tenseplus", "modifier"]
    mutation_types = ["that_this", "inversion","tenseplus", "modifier"]
    file_prefixs = []
    for dataset in datasets:
        for mutation_type in mutation_types:
            if mutation_type == "":
                file_prefix = '../../asset/Evaluation_Datasets/' + dataset + '/' + dataset
            else:
                file_prefix = '../../asset/Evaluation_Datasets/' + dataset + '_' + mutation_type+ '/' + dataset + '_' + mutation_type

            xml_file_name = file_prefix + '.data.xml'
            print(file_prefix)
            if not os.path.exists(file_prefix+'.csv'):
                generate_csv(xml_file_name)

            gold_key_file_name = file_prefix + '.gold.key.txt'
            csv_file_name = file_prefix + '.csv'
            csv_file_final_name = file_prefix + '_test_sent_cls_ws.csv'
            csv_token_cls_name = file_prefix + '_test_token_cls.csv'
            csv_cls_name = file_prefix + '_test_sent_cls.csv'
            if not os.path.exists(csv_file_final_name):
                generate_auxiliary_cls_ws(gold_key_file_name, csv_file_name, csv_file_final_name)
            if not os.path.exists(csv_token_cls_name):
                generate_auxiliary_token_cls(gold_key_file_name, csv_file_name, csv_token_cls_name)
            if not os.path.exists(csv_cls_name):
                generate_auxiliary_cls(gold_key_file_name, csv_file_name, csv_cls_name)
