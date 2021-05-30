# _*_ coding: utf-8 _*_
# Date: 2021-5-23
# Author: Qianqian Peng
# Reference:

import json
import os

def result_postproc(result_path,result_type,dict_path,pmid_txt):
    '''
    pmid_txt: a txt file, each line contains a pmid number
    result_path: list[str:dict_match_result_path,str:deep learning match result]
                    if result_type is 'conbined' else str
    '''
    dict_data = open(dict_path + '/all_name_id.txt','r').read().split('\n')
    if dict_data[-1] == '':
        dict_data.pop(-1)

    name2id = {}
    for one in dict_data:
        one = one.split('::')
        name = one[0].lower()
        _id = one[1]
        name2id[name] = _id
    id_int_map = {}
    with open(dict_path+'/mapping.txt','r') as f:
        for line in f:
           line = line.strip().split(' ')
           id_int_map[line[0]] = line[1]
    pmid_list = open(pmid_txt,'r').read().split('\n')
    if pmid_list[-1] == '':
        pmid_list.pop(-1)

    # for dict match
    if result_type is 'dict_match':
        file_list = os.listdir(result_path)
        pmid2labelInt = {}
        n = 0
        for file_name in file_list:
            with open(result_path+'/'+file_name,'r',encoding='utf-8') as f:
                pmid = file_name.split('.')[0]
                if pmid not in pmid_list:
                    continue
                n += 1
                print(n)
                label_int = []
                for line in f:
                    line = line.strip()
                    line = line.split('\t')
                    if len(line) == 1:
                        continue
                    mention = line[0]
                    names = line[1:]
                    elem_mention = set(mention)
                    coefficient = []
                    for one_name in names:
                        elem_one_name = set(one_name)
                        inter_elem = elem_one_name.intersection(elem_mention)
                        scores = len(inter_elem)/len(elem_one_name) + len(inter_elem)/len(elem_mention)
                        coefficient.append(scores)
                    if len(coefficient) == 1:
                        label_int.append(id_int_map[name2id[names[0].lower()]])
                    else:
                        max_score_idx = coefficient.index(max(coefficient))
                        label_int.append(id_int_map[name2id[names[max_score_idx].lower()]])
                label_int = list(set(label_int))
                if len(label_int) == 0:
                    label_int.append('29641')
                pmid2labelInt[pmid] = label_int
        pmids_extract = sorted(pmid2labelInt)
        f2 = open('../data/result/dict_BioASQ_format.txt','w',newline='\n')
        for one_pmid in pmids_extract:
            if len(pmid2labelInt[one_pmid]) == 0:
                pmid2labelInt[one_pmid].append('29641')
            out_line = ' '.join(pmid2labelInt[one_pmid]) + '\n'
            f2.write(out_line)


    # for MetaMap result
    if result_type is 'MetaMap':
        pmid2labelInt = {}
        with open(result_path,'r') as f:
            with open('../data/result/metamap_BioASQ_format.txt','w',newline='\n') as f2:
                for line in f:
                    print('*'*20)
                    line = json.loads(line)
                    pmid = line['pmid']
                    if pmid not in pmid_list:
                        continue
                    labels = line['meshMajor']
                    in_mesh_label = []
                    for one_label in labels:
                        one_label = one_label.lower()
                        if name2id.get(one_label) is not None:
                            in_mesh_label.append(str(id_int_map[name2id[one_label]]))
                    pmid2labelInt[pmid] = in_mesh_label
                pmids_extract = sorted(pmid2labelInt)

                for one_pmid in pmids_extract:
                    out_line = ' '.join(pmid2labelInt[one_pmid]) + '\n'
                    f2.write(out_line)

    # for classifier's result
    if result_type is 'classifier':
        with open('../data/result/classifier_BioASQ_format.txt','w',encoding='utf-8') as f2:
            file_list = os.listdir(result_path)
            pmid2labelInt = {}
            for file_name in file_list:
                with open(result_path+'/'+file_name,'r',encoding='utf-8') as f:
                    pmid = file_name.split('_')[0]
                    if pmid not in pmid_list:
                        continue
                    label_int = []
                    for line in f:
                        line = line.split('\t')
                        ann = line[1].split('::')
                        ann_label = ann[0]
                        ann_score = float(ann[2])
                        if ann_label != 'MS:None':
                            # print(ann_label)
                            if ann_score >=0.25:
                                label_int.append(str(id_int_map[ann_label]))
                    pmid2labelInt[pmid] = list(set(label_int))

            pmids_extract = sorted(pmid2labelInt)
            for one_pmid in pmids_extract:
                if len(pmid2labelInt[one_pmid]) == 0:
                    pmid2labelInt[one_pmid].append('29641')
                out_line = ' '.join(pmid2labelInt[one_pmid]) + '\n'
                f2.write(out_line)

    # for biencoder's result
    if result_type is 'biencoder':
        no = []
        with open('../data/result/biencoder_BioASQ_format.txt', 'w', encoding='utf-8') as f2:
            words_id_map = json.loads(open(dict_path+'/word_id_map.json','r').read())
            file_list = os.listdir(result_path)
            pmid2labelInt = {}
            for file_name in file_list:
                with open(result_path + '/' + file_name, 'r', encoding='utf-8') as f:
                    pmid = file_name.split('_')[0]
                    if pmid not in pmid_list:
                        continue
                    label_int = []
                    for line in f:
                        line = line.split('\t')
                        ann = line[1].split('::')
                        ann_name = ann[0]
                        ann_score = float(ann[1])
                        if ann_score >= 0.25:
                            # print(ann_score)
                            label_int.append(str(id_int_map[words_id_map[ann_name][0]]))
                    pmid2labelInt[pmid] = list(set(label_int))

            pmids_extract = sorted(pmid2labelInt)
            for one_pmid in pmids_extract:
                if len(pmid2labelInt[one_pmid]) == 0:
                    pmid2labelInt[one_pmid].append('29641')
                out_line = ' '.join(pmid2labelInt[one_pmid]) + '\n'
                f2.write(out_line)

    # for biTaskModel's result
    if result_type is 'biTask':
        no = []
        with open('../data/result/biTask_BioASQ_format.txt', 'w', encoding='utf-8') as f2:
            words_id_map = json.loads(open(dict_path + '/word_id_map.json', 'r').read())
            file_list = os.listdir(result_path)
            pmid2labelInt = {}
            for file_name in file_list:
                with open(result_path + '/' + file_name, 'r', encoding='utf-8') as f:
                    pmid = file_name.split('_')[0]
                    if pmid not in pmid_list:
                        continue
                    label_int = []
                    for line in f:
                        line = line.split('\t')
                        ann = line[1].split('::')
                        ann_label = ann[0]
                        ann_score = float(ann[2])
                        if ann_label != 'MS:None':
                            # print(ann_label)
                            if ann_score >= 0.25:
                                label_int.append(str(id_int_map[ann_label]))
                    pmid2labelInt[pmid] = list(set(label_int))

            pmids_extract = sorted(pmid2labelInt)
            for one_pmid in pmids_extract:
                if len(pmid2labelInt[one_pmid]) == 0:
                    pmid2labelInt[one_pmid].append('29641')
                out_line = ' '.join(pmid2labelInt[one_pmid]) + '\n'
                f2.write(out_line)

    # for combined result
    if result_type is 'combined':
        dict_result_path = result_path[0]
        classifier_result_path = result_path[1]
        dict_result_file_list = os.listdir(dict_result_path)
        classifier_result_file_list = os.listdir(classifier_result_path)
        dict_result_file_list = [filename for filename in dict_result_file_list
                                 if filename.split('.')[0] in pmid_list]
        classifier_result_file_list = [filename for filename in classifier_result_file_list
                                 if filename.split('_')[0] in pmid_list]
        pmid2word2label_score_dict = {}
        pmid2word2label_score_classifier = {}
        for filename in dict_result_file_list:
            pmid = filename.split('.')[0]
            word2label_score = {}
            with open(dict_result_path+'/'+filename,'r',encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    line = line.split('\t')
                    mention = line[0].lower()
                    if len(line) == 1:
                        word2label_score[mention] = ('MS:None',-100)
                    else:
                        names = line[1:]
                        elem_mention = set(mention)
                        coefficient = []
                        for one_name in names:
                            elem_one_name = set(one_name)
                            inter_elem = elem_one_name.intersection(elem_mention)
                            scores = len(inter_elem) / len(elem_one_name) + len(inter_elem) / len(elem_mention)
                            coefficient.append(scores)
                        if len(coefficient) == 1:
                            word2label_score[mention] = (name2id[names[0].lower()],coefficient[0]/2)
                        else:
                            max_score_idx = coefficient.index(max(coefficient))
                            word2label_score[mention] = (name2id[names[max_score_idx].lower()],coefficient[max_score_idx]/2)
            pmid2word2label_score_dict[pmid] = word2label_score

        for filename in classifier_result_file_list:
            pmid = filename.split('_')[0]
            word2label_score = {}
            with open(classifier_result_path+'/'+filename,'r',encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    line = line.split('\t')
                    mention = line[0].lower()
                    ann = line[1]
                    ann_label = ann.split('::')[0]
                    ann_score = float(ann.split('::')[2])
                    if ann_label == 'MS:None':
                        ann_score = -100
                    word2label_score[mention] = (ann_label,ann_score)
            pmid2word2label_score_classifier[pmid] = word2label_score
        pmid2labelInt = {}
        for pmid in pmid2word2label_score_dict.keys():
            word2label_score_dict = pmid2word2label_score_dict[pmid]
            word2label_score_classifier = pmid2word2label_score_classifier[pmid]
            labelInt = []
            print(pmid)
            for word in word2label_score_dict.keys():
                label_dict, score_dict = word2label_score_dict[word]
                label_class, score_class = word2label_score_classifier[word]
                if label_dict == label_class and score_class>=0.25:
                    labelInt.append(id_int_map[label_dict])
                else:
                    if word == label_dict:
                        labelInt.append(id_int_map[label_dict])
                    elif score_dict >= score_class and score_dict>= 0.25:
                        labelInt.append(id_int_map[label_dict])
                    else:
                        if score_class >= 0.25:
                            labelInt.append(id_int_map[label_class])
            labelInt = list(set(labelInt))
            pmid2labelInt[pmid] = labelInt
        pmids_extract = sorted(pmid2labelInt)
        with open('../data/result/combined_BioASQ_format.txt', 'w', encoding='utf-8') as f:
            for one_pmid in pmids_extract:
                out_line = ' '.join(pmid2labelInt[one_pmid]) + '\n'
                f.write(out_line)

def map_gold_result(gold_path):

    pass




if __name__ == '__main__':
    result_postproc(['../data/test_data/result_3/','../data/test_data/classifier_result_selected/'],'combined','../dict/','../data/result/pmid.txt')
    # result_postproc('../data/test_data/biencoder_result_selected/','biencoder','../dict/','../data/result/pmid.txt')
    # map_gold_result('../data/')



