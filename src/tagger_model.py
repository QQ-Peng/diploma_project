# Date: 2021-5-9
# Author: Qianqian Peng

import torch
from transformers import BertModel,BertTokenizer
import os, json
from classifier_train import BiTaskBert
from Biencoder_train import BiEncoder
from config import ConfigClassifier, ConfigBiEncoder
from utlis import *
import time

def evaluate_classifier(config):
    with torch.no_grad():
        topk=2
        tokenizer = BertTokenizer.from_pretrained(config.bert_path)
        label = open(config.dict_path + '/lable.vocab', 'r').read().split('\n')
        if label[-1] == '':
            label.pop(-1)
        label2idx = label_2_index(label)
        idx2label = {idx:label for label,idx in label2idx.items()}
        id2word = json.loads(open(config.dict_path + '/id_word_map.json', 'r').read())
        model = BiTaskBert(config)
        model.load_state_dict(torch.load(config.evaluate_model))
        model.to(config.device)
        file_list = os.listdir(config.evaluate_data)
        if not os.path.exists(config.out_path):
            os.mkdir(config.out_path)

        n_file = 0
        for file in file_list:
            start = time.time()
            pmid = file.split('.')[0]
            out_file = config.out_path + '/' + pmid + '_result.cand'
            if os.path.exists(out_file):
                print(out_file+" exist")
                continue
            out_file = open(out_file,'w',encoding='utf-8',newline='\n')
            candidates = open(config.evaluate_data+file,'r',encoding='utf-8').readlines()
            candidates = [cand.strip() for cand in candidates]
            print("*"*20)
            print(pmid)
            print(len(candidates))
            print(candidates)
            n = 0
            while n < len(candidates):
                cand_batch = [cand.lower() for cand in candidates[n:n+200]]
                inputs = tokenizer.batch_encode_plus(cand_batch,padding=True)
                input_ids = torch.tensor(inputs['input_ids']).to(config.device)
                attention_mask = torch.tensor(inputs['attention_mask']).to(config.device)
                token_type_ids = torch.tensor(inputs['token_type_ids']).to(config.device)
                prob = model.predict(input_ids,attention_mask,token_type_ids)
                pred_score, pred_index = prob.topk(topk)
                cand_ori = candidates[n:n + 200]
                for i in range(len(cand_ori)):
                    out_file.write(cand_ori[i]+'\t')
                    for j in range(topk):
                        _id = idx2label[pred_index[i,j].item()]
                        mesh_term = id2word[_id][0] if _id != 'MS:None' else 'None'
                        if j < topk-1:
                            out_file.write(_id + '::' + mesh_term + '::' + str(pred_score[i,j].item()) +'\t')
                        else:
                            out_file.write(_id + '::' + mesh_term + '::' + str(pred_score[i, j].item()) + '\n')

                n += 200
            out_file.close()
            # print(prob.shape)
            '''
                prob=torch.tensor([[0.2,0.1,0.7],[0.6,0.1,0.3]])
                torch.return_types.topk(
values=tensor([[0.7000, 0.2000],
        [0.6000, 0.3000]]),
indices=tensor([[2, 0],
        [0, 2]]))
                '''

            n_file += 1
            end = time.time()
            print("process {}/{}, duration: {}".format(n_file,len(file_list),end-start))


def evaluate_biencoder(config):
    with torch.no_grad():
                word2id = json.loads(open(config.dict_path + '/word_id_map.json').read())
                words = list(word2id.keys())
                print(words[0:20])
                print(len(words))
                tokenizer = BertTokenizer.from_pretrained(config.bert_path)
                model = BiEncoder(config)
                model.load_state_dict(torch.load(config.evaluate_model))
                model.to(config.device)
                # model.EntityEncoder.to('cuda')
                if not config.pre_compute_words_embed:
                    print("********compute words embedding********")
                    words_embed = []
                    cur_n = 0
                    while cur_n < len(words):
                        words_inputs = tokenizer.batch_encode_plus(words[cur_n:cur_n+128], padding=True)
                        words_input_ids = torch.tensor(words_inputs['input_ids']).to('cuda')
                        words_attention_mask = torch.tensor(words_inputs['attention_mask']).to('cuda')
                        words_token_type_ids = torch.tensor(words_inputs['token_type_ids']).to('cuda')
                        out = model.EntityEncoder(words_input_ids, words_attention_mask, words_token_type_ids)
                        embed_batch = out[1]
                        embed_batch = model.EntityLinearLayer(embed_batch)
                        words_embed.append(embed_batch)
                        cur_n += 128
                        print(cur_n)
                    words_embed=torch.cat(words_embed)
                    torch.save(words_embed,config.words_embed)
                else:
                    print("********load words embedding********")
                    words_embed = torch.load(config.words_embed)
                    print("words_embed's shape: ",words_embed.shape)
                file_list = os.listdir(config.evaluate_data)
                if not os.path.exists(config.out_path):
                    os.mkdir(config.out_path)
                n_file = 0

                for file in file_list:
                        start = time.time()
                        pmid = file.split('.')[0]
                        out_file = config.out_path + '/' + pmid + '_result.cand'
                        if os.path.exists(out_file):
                            print(out_file + " exist")
                            continue
                        out_file = open(out_file, 'w', encoding='utf-8', newline='\n')
                        candidates = open(config.evaluate_data+file,'r',encoding='utf-8').readlines()
                        candidates = [cand.strip() for cand in candidates]
                        print("*"*20)
                        print(pmid)
                        print(len(candidates))
                        print(candidates)
                        n = 0
                        step = 50
                        while n < len(candidates):
                                inputs = tokenizer.batch_encode_plus(candidates[n:n+step],padding=True)
                                input_ids = torch.tensor(inputs['input_ids']).to(config.device)
                                attention_mask = torch.tensor(inputs['attention_mask']).to(config.device)
                                token_type_ids = torch.tensor(inputs['token_type_ids']).to(config.device)
                                mention_embed = model.MentionEncoder(input_ids,attention_mask,token_type_ids)[1]
                                mention_embed = model.MentionLinearLayer(mention_embed)
                                print("mention embed's shape: ",mention_embed.shape)
                                scores = mention_embed@words_embed.t()
                                scores = torch.softmax(scores,dim=1)
                                pred_score,pred_idx = scores.topk(1)
                                pred_idx = pred_idx.squeeze(1).tolist()
                                pred_score = pred_score.squeeze(1).tolist()
                                cand_ori = candidates[n:n + step]
                                for i in range(len(cand_ori)):
                                    out_file.write(cand_ori[i] + '\t'+words[pred_idx[i]]+'::'+str(pred_score[i])+'\n')

                                n += step
                        out_file.close()


def evaluate_biTaskModel(config):
    with torch.no_grad():
        topk = 2
        word2id = json.loads(open(config.dict_path + '/word_id_map.json', 'r').read())
        id2word = json.loads(open(config.dict_path + '/id_word_map.json', 'r').read())
        all_id=[]
        for i,j in word2id.items():
            all_id += j
        ori_id2name = open(config.dict_path+'/id2name.txt','r').read().split('\n')
        if ori_id2name[-1] == '':
            ori_id2name.pop(-1)
        for one in ori_id2name:
            one = one.split('::')
            _id = 'MS:' + one[0]
            syns = one[1:]
            if _id not in all_id:
                for one_syn in syns:
                    if word2id.get(one_syn) is None:
                        word2id[one_syn] = [_id]
                    else:
                        word2id[one_syn].append(_id)
        for _word, _id in word2id.items():
            word2id[_word] = _id[0]
        del word2id['All']
        word2id['None'] = 'MS:None'

        label = open(config.dict_path + '/lable.vocab', 'r').read().split('\n')
        if label[-1] == '':
            label.pop(-1)
        label2idx = label_2_index(label)
        idx2label = {idx: label for label, idx in label2idx.items()}
        words = list(word2id.keys())
        # print("names: ",words)
        print(len(words))
        tokenizer = BertTokenizer.from_pretrained(config.bert_path)
        model = BiTaskBert(config)
        model.load_state_dict(torch.load(config.evaluate_model))
        model.to(config.device)
        if not config.pre_compute_words_embed:
            print("********compute words embedding********")
            words_embed = []
            cur_n = 0
            while cur_n < len(words):
                words_inputs = tokenizer.batch_encode_plus(words[cur_n:cur_n+128], padding=True)
                words_input_ids = torch.tensor(words_inputs['input_ids']).to('cuda')
                words_attention_mask = torch.tensor(words_inputs['attention_mask']).to('cuda')
                words_token_type_ids = torch.tensor(words_inputs['token_type_ids']).to('cuda')
                out = model.BertEncoder(words_input_ids, words_attention_mask, words_token_type_ids)
                embed_batch = out[1]
                embed_batch = model.linking_layer_name(embed_batch)
                words_embed.append(embed_batch)
                cur_n += 128
                print(cur_n)
            words_embed=torch.cat(words_embed)
            torch.save(words_embed,config.words_embed)
        else:
            print("********load words embedding********")
            words_embed = torch.load(config.words_embed)
            print("words_embed's shape: ",words_embed.shape)
        file_list = os.listdir(config.evaluate_data)
        if not os.path.exists(config.out_path):
            os.mkdir(config.out_path)
        n_file = 0

        for file in file_list:
                start = time.time()
                pmid = file.split('.')[0]
                out_file = config.out_path + '/' + pmid + '_result.cand'
                if os.path.exists(out_file):
                    print(out_file + " exist")
                    continue
                out_file = open(out_file, 'w', encoding='utf-8', newline='\n')
                candidates = open(config.evaluate_data+file,'r',encoding='utf-8').readlines()
                # print(candidates)
                candidates = [cand.strip() for cand in candidates]
                # print(candidates)
                # print("*"*20)
                # print(pmid)
                # print(len(candidates))
                # print(candidates)
                n = 0
                step = 200
                while n < len(candidates):
                        inputs = tokenizer.batch_encode_plus(candidates[n:n+step],padding=True)
                        input_ids = torch.tensor(inputs['input_ids']).to(config.device)
                        attention_mask = torch.tensor(inputs['attention_mask']).to(config.device)
                        token_type_ids = torch.tensor(inputs['token_type_ids']).to(config.device)
                        mention_embed = model.BertEncoder(input_ids,attention_mask,token_type_ids)[1]
                        pred_prob = torch.softmax(model.classifier_layer(mention_embed),dim=1)
                        mention_embed_link = model.linking_layer_mention(mention_embed)
                        print("mention embed's shape: ",mention_embed.shape)
                        scores = mention_embed_link@words_embed.t()
                        scores = torch.softmax(scores,dim=1)
                        scores = scores.to('cpu')
                        pred_prob = pred_prob.to('cpu')
                        cand_ori = candidates[n:n + step]
                        link_scores = []
                        for i in range(scores.shape[0]):
                            labelIdx2score={}
                            for j in range(scores.shape[1]):
                                labelIdx = label2idx[word2id[words[j]]]
                                if labelIdx2score.get(labelIdx) is None:
                                    labelIdx2score[labelIdx] = [scores[i,j].item()]
                                else:
                                    labelIdx2score[labelIdx].append(scores[i,j].item())
                            for idx,s_list in labelIdx2score.items():
                                labelIdx2score[idx] = max(s_list)

                            max_score = [labelIdx2score[idx] for idx in range(len(labelIdx2score))]
                            link_scores.append(torch.tensor([max_score]))
                        link_scores = torch.cat(link_scores)
                        total_scores = 0.3*link_scores+ 0.7*pred_prob
                        pred_score,pred_index = total_scores.topk(topk)
                        for i in range(len(cand_ori)):
                            out_file.write(cand_ori[i] + '\t')
                            for j in range(topk):
                                _id = idx2label[pred_index[i, j].item()]
                                mesh_term = id2word[_id][0] if _id != 'MS:None' else 'None'
                                if j < topk - 1:
                                    out_file.write(
                                        _id + '::' + mesh_term + '::' + str(pred_score[i, j].item()) + '\t')
                                else:
                                    out_file.write(
                                        _id + '::' + mesh_term + '::' + str(pred_score[i, j].item()) + '\n')

                            # out_file.write(cand_ori[i] + '\t'+words[pred_idx[i]]+'::'+str(pred_score[i])+'\n')

                        n += step
                out_file.close()

def evaluate_biTaskModel_fast(config):
    with torch.no_grad():
        topk = 2
        ori_id2name = open(config.dict_path+'/id2name.txt','r').read().split('\n')
        ori_id2name.pop(0)
        if ori_id2name[-1] == '':
            ori_id2name.pop(-1)
        id2name = {}
        for one in ori_id2name:
            one = one.split('::')
            _id = 'MS:' + one[0]
            name= one[1]
            id2name[_id] = name
        id2name['MS:None'] = 'None'
        label = open(config.dict_path + '/lable.vocab', 'r').read().split('\n')
        if label[-1] == '':
            label.pop(-1)
        label2idx = label_2_index(label)
        idx2label = {idx: label for label, idx in label2idx.items()}
        words = list(id2name.values())
        # print("names: ",words)
        # print(len(words))
        tokenizer = BertTokenizer.from_pretrained(config.bert_path)
        model = BiTaskBert(config)
        model.load_state_dict(torch.load(config.evaluate_model))
        model.to(config.device)
        if not config.pre_compute_words_embed:
            print("********compute words embedding********")
            words_embed = []
            cur_n = 0
            while cur_n < len(words):
                words_inputs = tokenizer.batch_encode_plus(words[cur_n:cur_n+128], padding=True)
                words_input_ids = torch.tensor(words_inputs['input_ids']).to('cuda')
                words_attention_mask = torch.tensor(words_inputs['attention_mask']).to('cuda')
                words_token_type_ids = torch.tensor(words_inputs['token_type_ids']).to('cuda')
                out = model.BertEncoder(words_input_ids, words_attention_mask, words_token_type_ids)
                embed_batch = out[1]
                embed_batch = model.linking_layer_name(embed_batch)
                words_embed.append(embed_batch)
                cur_n += 128
                print(cur_n)
            words_embed=torch.cat(words_embed)
            torch.save(words_embed,config.words_embed)
        else:
            print("********load words embedding********")
            words_embed = torch.load(config.words_embed)
            print("words_embed's shape: ",words_embed.shape)
        file_list = os.listdir(config.evaluate_data)
        if not os.path.exists(config.out_path):
            os.mkdir(config.out_path)
        n_file = 0

        for file in file_list:
                start = time.time()
                pmid = file.split('.')[0]
                out_file = config.out_path + '/' + pmid + '_result.cand'
                if os.path.exists(out_file):
                    print(out_file + " exist")
                    continue
                out_file = open(out_file, 'w', encoding='utf-8', newline='\n')
                candidates = open(config.evaluate_data+file,'r',encoding='utf-8').readlines()
                # print(candidates)
                candidates = [cand.strip() for cand in candidates]
                # print(candidates)
                # print("*"*20)
                # print(pmid)
                # print(len(candidates))
                # print(candidates)
                n = 0
                step = 200
                while n < len(candidates):
                        inputs = tokenizer.batch_encode_plus(candidates[n:n+step],padding=True)
                        input_ids = torch.tensor(inputs['input_ids']).to(config.device)
                        attention_mask = torch.tensor(inputs['attention_mask']).to(config.device)
                        token_type_ids = torch.tensor(inputs['token_type_ids']).to(config.device)
                        mention_embed = model.BertEncoder(input_ids,attention_mask,token_type_ids)[1]
                        pred_prob = torch.softmax(model.classifier_layer(mention_embed),dim=1)
                        mention_embed_link = model.linking_layer_mention(mention_embed)
                        print("mention embed's shape: ",mention_embed.shape)
                        link_scores = mention_embed_link@words_embed.t()
                        link_scores = torch.softmax(link_scores,dim=1)
                        total_scores = 0.3*link_scores+ 0.7*pred_prob
                        pred_score,pred_index = total_scores.topk(topk)
                        cand_ori = candidates[n:n + step]
                        for i in range(len(cand_ori)):
                            out_file.write(cand_ori[i] + '\t')
                            for j in range(topk):
                                _id = idx2label[pred_index[i, j].item()]
                                mesh_term = id2name[_id] if _id != 'MS:None' else 'None'
                                if j < topk - 1:
                                    out_file.write(
                                        _id + '::' + mesh_term + '::' + str(pred_score[i, j].item()) + '\t')
                                else:
                                    out_file.write(
                                        _id + '::' + mesh_term + '::' + str(pred_score[i, j].item()) + '\n')

                            # out_file.write(cand_ori[i] + '\t'+words[pred_idx[i]]+'::'+str(pred_score[i])+'\n')

                        n += step
                out_file.close()

# candidates = open('./data/test_data/Articles_PubTator_candidate/12806.PubTator.cand','r',encoding='utf-8').readlines()
# config = ConfigBiEncoder()
# wordembed=evaluate_biencoder(config)

if __name__== "__main__":
    # config = ConfigClassifier(True)
    # evaluate_classifier(config)
    # config = ConfigBiEncoder()
    # evaluate_biencoder(config)
    config = ConfigClassifier(False)
    wordembed = evaluate_biTaskModel_fast(config)