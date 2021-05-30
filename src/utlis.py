# Date: 2021-4-13
# Author: Qianqian Peng

import random
import os, json
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
from text_processor import sent_tokenize, word_tokenize, POS_tag
import string

def read_train_data(data_path,id2word,label):
    data = open(data_path,'r',encoding='utf-8').read().split('\n\n')
    data_num = len(data)
    label_num = len(label)
    print("data number: {}".format(data_num))
    i = 0
    for _ in range(data_num):
        i += 1
        # print('{}/{}'.format(i,data_num))
        one_data = data.pop(0)
        data.append(one_data.split('\n')[0].split('\t'))
        if data[-1][0] == 'MS:None':
            # print('here1')
            rand_idx = random.randint(0,label_num-1)
            while label[rand_idx] == 'MS:None':
                rand_idx = random.randint(0, label_num - 1)
                # print(label[rand_idx])
            # print(label[rand_idx])
            data[-1].append(id2word[label[rand_idx]][0])
            data[-1].append(0)
            if len(data[-1][1].split(' '))>10 or len(data[-1][2].split(' '))>10:
                data.pop(-1)
        else:
            data[-1].append(id2word[data[-1][0]][-1])
            data[-1].append(1)
            if len(data[-1][1].split(' ')) > 10 or len(data[-1][2].split(' ')) > 10:
                data.pop(-1)
    return data

def label_2_index(label_list):
    label2index = {}
    for i in range(len(label_list)):
        label2index[label_list[i]] = i
    return label2index

def convert_obo_dict(ontology_path, save=False, path='.'):
    ontology = open(ontology_path, 'r', encoding='utf-8').read().split('[Term]')
    ontology.pop(0)
    ontology = [record.strip().split('\n') for record in ontology]
    id2name = {}
    name2id = {}
    for record in ontology:
        _id = None
        name = []
        for attr in record:
            if attr.startswith('id: MS:'):
                _id = attr[len('id: MS:'):]
            if attr.startswith('name: '):
                name.append(attr[len('name: '):])
            if attr.startswith('synonym: '):
                syn = attr[len('synonym: '):]
                syn = syn.split('EXACT')[0][:-1]
                name.append(syn)
        id2name[_id] = name
        for syn in name:
            if name2id.get(syn) is None:
                name2id[syn] = [_id]
            else:
                name2id[syn].append(_id)
    if save:
        with open(path+'/'+'id2name.txt','w',encoding='utf-8',newline='\n') as f:
            for _id, names in id2name.items():
                name_num = len(names)
                for _ in range(name_num):
                    name = names.pop(0)
                    if name[0] == '"' and name[-1] == '"':
                        name = name[1:-1]
                    names.append(name)
                f.write(_id+'::'+'::'.join(names)+'\n')
        with open(path+'/'+'name2id.txt','w',encoding='utf-8',newline='\n') as f:
            for name, ids in name2id.items():
                if name[0] == '"' and name[-1] == '"':
                    name = name[1:-1]
                f.write(name+'::'+'::'.join(ids)+'\n')

    return id2name, name2id

def generate_candidate(tokens:list, detokenizer, max_len=8):
    candidates = []
    for i in range(1,max_len+1):
        for pos in range(len(tokens)-i+1):
            candidates.append(detokenizer(tokens[pos:pos+i]))
    return candidates

def generate_candidate_to_file(filedir, outdir,sentence_tokenizer, token_tokenizer, detokenizer, max_len=8):
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    for file in os.listdir(filedir):
        text = open(file_dir+file, 'r', encoding='utf-8').read()
        sentences = sent_tokenize(sentence_tokenizer, text)
        sent_num = len(sentences)
        for _ in range(sent_num):
            sent = sentences.pop(0)
            sent = sent if '|t|' not in sent and '|a|' not in sent else \
                sent.split('|t|')[1] if '|t|' in sent else sent.split('|a|')[1]
            sentences.append(sent)
        tokens = word_tokenize(token_tokenizer, sentences)
        candidates = [generate_candidate(sent, detokenizer) for sent in tokens]

        with open(out_dir+'/'+file+'.cand','w',encoding='utf-8',newline='\n') as f:
            global n
            n += 1
            print("process {}, {}".format(file,n))
            punc = list(string.punctuation)
            for sent in candidates:
                for cand in sent:
                    try:
                        tokens = word_tokenize(token_tokenizer, cand)[0]
                        begin_token = tokens[0]
                        end_token = tokens[-1]
                        pos_begin = POS_tag(nltk.pos_tag, begin_token)[0][1]
                        pos_end = POS_tag(nltk.pos_tag, end_token)[0][1]
                        if begin_token in punc or pos_begin in ['IN', 'CC', 'DT'] or \
                                end_token in punc or pos_end in ['IN', 'CC', 'DT']:
                            continue
                        else:
                            f.write(cand+'\n')
                    except:
                        f.write(cand + '\n')

def convert_goldTerm_to_goldInt(data_path, pmid_txt,dict_path):
    dict_data = open(dict_path + '/all_name_id.txt', 'r').read().split('\n')
    if dict_data[-1] == '':
        dict_data.pop(-1)
    name2id = {}
    for one in dict_data:
        one = one.split('::')
        name = one[0].lower()
        _id = one[1]
        name2id[name] = _id
    id_int_map = {}
    with open(dict_path + '/mapping.txt', 'r') as f:
        for line in f:
            line = line.strip().split(' ')
            id_int_map[line[0]] = line[1]

    pmid_list = open(pmid_txt,'r').read().split('\n')
    if pmid_list[-1] == '':
        pmid_list.pop(-1)

    with open(data_path,'r',encoding='iso8859') as f:
        with open('../data/result/gold_BioASQ_format.txt','w',newline='\n') as f2:
            n = 0
            pmid2labelInt = {}
            for line in f:
                n += 1
                if n % 10000 == 0:
                    print(n)
                line = line.strip()
                if line[-1] == ',':
                    line = line[:-1]
                elif line[-1] == '}':
                    line = line[:-2]
                else:
                    continue
                record = json.loads(line)
                pmid = record['pmid']
                if pmid not in pmid_list:
                    continue
                mesh = record['meshMajor']
                num = len(mesh)
                label_idx = []
                for i in range(num):
                    one_mesh = mesh[i]
                    one_mesh = one_mesh.lower()
                    if name2id.get(one_mesh) is not None:
                        label_idx.append(id_int_map[name2id[one_mesh]])
                label_idx = list(set(label_idx))
                if len(label_idx) == 0:
                    label_idx.append('29641')
                pmid2labelInt[pmid] = label_idx
                pmid_extract = sorted(pmid2labelInt)
            for one_pmid in pmid_extract:
                out_line = " ".join(pmid2labelInt[one_pmid]) + '\n'
                f2.write(out_line)




n=0
if __name__ == "__main__":

    run_generate = False
    run_convert_obo = False
    save_mapping = False
    convert = False

    if run_generate:
        detokenizer = TreebankWordDetokenizer().detokenize
        file_dir = '../data/test_data/Articles_PubTator/'
        out_dir = '../data/test_data/Articles_PubTator_candidate/'
        generate_candidate_to_file(file_dir,out_dir,nltk.sent_tokenize,nltk.word_tokenize,detokenizer)
    if run_convert_obo:
        obo_path = '../ontology/mesh_addsyn.obo'
        out_path = '../dict/'
        id2name,name2id = convert_obo_dict(obo_path,False,out_path)

    if save_mapping:
        label = open('../dict/lable.vocab', 'r').read().split('\n')
        if label[-1] == '':
            label.pop(-1)
        label2idx = label_2_index(label)
        out = open('../dict/mapping.txt','w',newline='\n')
        for label, idx in label2idx.items():
            out.write(label+' '+str(idx+1)+'\n')
        out.close()
    if convert:
        convert_goldTerm_to_goldInt('F:/Data/allMeSH_2020/allMeSH_2020.json','../data/result/pmid.txt','../dict/')
        # java -Xmx10G -cp $CLASSPATH:./BioASQEvaluation/dist/BioASQEvaluation.jar evaluation.Evaluator /mnt/f/PYProject/diploma_project//data/result/gold_BioASQ_format.txt /mnt/f/PYProject/diploma_project/data/result/classifier_BioASQ_format.txt
