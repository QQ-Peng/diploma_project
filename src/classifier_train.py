# _*_ coding: utf-8 _*_
# Date: 2021-4-6
# Author: Qianqian Peng
# Reference:

import torch.nn as nn
import torch, json
from transformers import BertModel, BertTokenizer
import torch.tensor as Tensor
import copy
from config import ConfigClassifier
from utlis import *
import argparse
import os

class BiTaskBert(nn.Module):
    def __init__(self,config):
        super(BiTaskBert,self).__init__()
        self.BertEncoder = BertModel.from_pretrained(config.bert_path)
        self.classifier_layer = nn.Linear(config.bert_out_dim,config.class_num)
        self.linking_layer_name = nn.Linear(config.bert_out_dim,config.bert_out_dim)
        self.linking_layer_mention = nn.Linear(config.bert_out_dim, config.bert_out_dim)
    def forward(self,clf_input_ids,clf_attention_mask,clf_token_type_ids,
                lnk_input_ids,lnk_attention_mask,lnk_token_type_ids,clf_label,lnk_label):
        '''
        clf: classification
        lnk: linking
        '''
        clf_embedding = self.BertEncoder(clf_input_ids,clf_attention_mask,clf_token_type_ids)[1]
        lnk_embedding_name = self.linking_layer_name(clf_embedding)
        clf_prob = torch.softmax(self.classifier_layer(clf_embedding),dim=1)
        # print("lnk_embedding_name.shape: ",lnk_embedding_name.shape)
        lnk_embedding_mention = self.BertEncoder(lnk_input_ids,lnk_attention_mask,lnk_token_type_ids)[1]
        lnk_embedding_mention = self.linking_layer_mention(lnk_embedding_mention)
        # print("lnk_embedding_mention.shape: ",lnk_embedding_mention.shape)
        clf_fct = nn.BCELoss(reduction='mean')
        lnk_fct = nn.BCEWithLogitsLoss(reduction='mean')
        clf_loss = clf_fct(clf_prob,clf_label.float())
        lnk_score = (lnk_embedding_name@lnk_embedding_mention.t()).diag()
        lnk_loss = lnk_fct(lnk_score.unsqueeze(1),lnk_label.float().unsqueeze(1))
        return clf_loss,lnk_loss,clf_prob
    def predict(self,input_ids,attention_mask,token_type_ids):
        clf_embedding = self.BertEncoder(input_ids, attention_mask, token_type_ids)[1]
        clf_prob = torch.softmax(self.classifier_layer(clf_embedding), dim=1)
        return clf_prob






def train(config):
    if config.device is 'cuda':
        torch.cuda.set_device(config.device_n)
    id2word = json.loads(open(config.dict_path + '/id_word_map.json', 'r').read())
    word2id = json.loads(open(config.dict_path+'/word_id_map.json', 'r').read())
    label = open(config.dict_path+'/lable.vocab', 'r').read().split('\n')
    if label[-1] == '':
        label.pop(-1)
    print('read and build training data...')
    data = read_train_data(config.train_data, id2word, label)
    print('read and build training data done...')

    label2idx = label_2_index(label)
    config.class_num = len(label2idx)
    device = config.device
    print('load the pretrained model...')
    tokenizer = BertTokenizer.from_pretrained(config.bert_path)
    model = BiTaskBert(config)
    print('load the pretrained model done...')
    # if torch.cuda.device_count() > 1:
    #     print('Use', torch.cuda.device_count(), 'gpus')
    #     model = nn.DataParallel(model)
    model.to(device)
    print('fine tune the model...')
    optimizer = torch.optim.Adam(params=model.parameters(),lr=config.lr)
    for epoch in range(config.n_epoch):
        n = 0
        random.shuffle(data)
        batch_size = config.batch_size
        data_num = len(data)

        n_batch = 0
        while n < data_num-1:
            optimizer.zero_grad()

            n_batch += 1
            batch_data = data[n:n+batch_size]
            label_batch = [i[0] for i in batch_data]
            name_batch = [i[1] for i in batch_data]
            synonym_batch = [i[2] for i in batch_data]
            linking_label_batch = [i[3] for i in batch_data]
            linking_label_batch = Tensor(linking_label_batch,device=device)
            # print(linking_label_batch)
            classification_input = tokenizer.batch_encode_plus(name_batch,padding=True,max_length=128)
            classification_input_ids = Tensor(classification_input['input_ids'],device=device)
            classification_attention_mask = Tensor(classification_input['attention_mask'],device=device)
            classification_token_type_ids = Tensor(classification_input['token_type_ids'], device=device)
            linking_input = tokenizer.batch_encode_plus(synonym_batch,padding=True,max_length=128)
            linking_input_ids = Tensor(linking_input['input_ids'],device=device)
            linking_attention_mask = Tensor(linking_input['attention_mask'],device=device)
            linking_token_type_ids = Tensor(linking_input['token_type_ids'],device=device)
            classification_label_idx = [label2idx[i] for i in label_batch]
            classification_label = [[0]*config.class_num for _ in range(len(label_batch))]
            if classification_input_ids.shape[1]>200 or linking_input_ids.shape[1]>200:
                n += batch_size
                continue
            for i in range(len(label_batch)):
                # print(classification_label_idx[i])
                classification_label[i][classification_label_idx[i]] = 1
            classification_label = Tensor(classification_label,device=device)
            clf_loss,lnk_loss,clf_prob = model(classification_input_ids,classification_attention_mask,classification_token_type_ids,
                  linking_input_ids,linking_attention_mask,linking_token_type_ids,
                  classification_label,linking_label_batch)
            n += batch_size
            if config.onlyclf:
                total_loss = clf_loss
            else:
                total_loss = 0.7*clf_loss + 0.3*lnk_loss
            if n_batch % 8 == 0:
                print("*" * 10, 'epoch: ', epoch+1, ' batch: ', n_batch, "*" * 10)
                if not config.onlyclf:
                   print('cls_loss: {}'.format(clf_loss))
                   print('lnk_loss: {}'.format(lnk_loss))
                print('total_loss: {}'.format(total_loss))
            total_loss.backward()
            optimizer.step()
        if (epoch+1) % config.save_step == 0:
            print('save the model...')
            if not os.path.exists(config.model_save_path):
                os.mkdir(config.model_save_path)
            torch.save(model.state_dict(), config.model_save_path+'/'+'model'+'_'+str(epoch+1)+'.bin')
            print('save the model done...')

    print('fine tune the model done...')

def evaluate(config):
    device = config.device
    label = open(config.dict_path + '/lable.vocab', 'r').read().split('\n')
    label2idx = label_2_index(label)
    idx2label = {idx: label for label, idx in label2idx.items()}
    config.class_num = len(label2idx)
    model = BiTaskBert(config)
    model.load_state_dict(torch.load(config.evaluate_model))
    model = model.to(device)
    model.eval()
    right_pred = 0
    all = 0
    with open(config.evaluate_data,'r',encoding='utf-8') as f:
        for line in f:
            all += 1
            line = json.loads(line)
            mention = line['mention']
            gold_label = line['hpoid']
            input = model.tokenizer.encode_plus(mention)
            input_ids = Tensor(input['input_ids'],device=device).unsqueeze(0)
            attention_mask = Tensor(input['attention_mask'],device=device).unsqueeze(0)
            token_type_ids = Tensor(input['token_type_ids'],device=device).unsqueeze(0)
            # print('input_ids.shape: ',input_ids.shape)
            clf_prob = model.predict(input_ids,attention_mask,token_type_ids)
            _,idx = clf_prob.topk(1)
            idx = idx.squeeze(0).tolist()
            # print(idx)
            # print(type(idx))
            # print(idx2label)
            pred_label = [idx2label[i] for i in idx]
            if gold_label in pred_label:
                right_pred += 1
            print('all: {}'.format(right_pred/all))



    pass
if __name__ == '__main__':
    # parser = argparse.ArgumentParser(
    #     description='build weak training corpus, python build_dict.py -i infile -o outpath')
    # parser.add_argument('--infolder', '-i', help="input folder path", default='../example/input/')
    # parser.add_argument('--outfolder', '-o', help="output folder path", default='../example/output/')
    #
    # args = parser.parse_args()
    # args.outfolder = args.outfolder + '/'
    # args.infolder = args.infolder + '/'

    config = ConfigClassifier()
    train(config)
    #evaluate(config)
