# Date: 2021-4-13
# Author: Qianqian Peng

from config import ConfigBiEncoder
from transformers import BertModel,BertTokenizer
import torch.nn as nn
import torch.tensor as Tensor
import torch
from utlis import *
import os
import json
class BiEncoder(nn.Module):
    def __init__(self,config):
        super(BiEncoder,self).__init__()
        self.EntityEncoder = BertModel.from_pretrained(config.bert_path)
        self.MentionEncoder = BertModel.from_pretrained(config.bert_path)
        self.EntityLinearLayer = nn.Linear(config.bert_out_dim,config.bert_out_dim)
        self.MentionLinearLayer = nn.Linear(config.bert_out_dim,config.bert_out_dim)

    def forward(self,entity_input_ids,entity_attention_mask,entity_token_type_ids,
                mention_input_ids,mention_attention_mask,mention_token_type_ids,label):

        entity_embedding = self.EntityEncoder(entity_input_ids,entity_attention_mask,entity_token_type_ids)[1]
        mention_embedding = self.MentionEncoder(mention_input_ids, mention_attention_mask, mention_token_type_ids)[1]
        # print("entity embedding: ", entity_embedding)
        # print("mention embedding: ", mention_embedding)
        entity_embedding = self.EntityLinearLayer(entity_embedding)
        mention_embedding = self.MentionLinearLayer(mention_embedding)
        loss_fct = nn.BCEWithLogitsLoss(reduction='mean')
        # print("entity embedding: ",entity_embedding)
        # print("mention embedding: ",mention_embedding)
        lnk_score = (entity_embedding@mention_embedding.t()).diag().unsqueeze(-1)
        # print("lnk_score: ",lnk_score)
        # print("label: ",label)
        # print("lnk_score shape: ", lnk_score.shape)
        # print("label shape: ", label.shape)
        loss = loss_fct(lnk_score,label.float().unsqueeze(1))
        return entity_embedding, mention_embedding, loss


def train(config):
    if config.device is 'cuda':
        torch.cuda.set_device(config.device_n)
    id2word = json.loads(open(config.dict_path + '/id_word_map.json', 'r').read())
    word2id = json.loads(open(config.dict_path + '/word_id_map.json', 'r').read())
    label = open(config.dict_path + '/lable.vocab', 'r').read().split('\n')
    if label[-1] == '':
        label.pop(-1)
    print('read and build training data...')
    data = read_train_data(config.train_data, id2word, label)
    pos=0
    neg=0
    for i in data:
        if i[3]==0:
            neg+=1
        else:
            pos+=1
    print("pos: ",pos)
    print("neg: ",neg)

    # test code below
    tokenizer = BertTokenizer.from_pretrained(config.bert_path)
    n = 0
    all_name_len = []
    all_syn_len = []
    for i in data:
        all_name_len.append(len(tokenizer.encode(i[1])))
        all_syn_len.append(len(tokenizer.encode(i[2])))
    print(max(all_name_len),' ',max(all_syn_len))
    return 0
    # test code above
    print('read and build training data done...')
    label2idx = label_2_index(label)
    config.class_num = len(label2idx)
    device = config.device
    print('load the pretrained model...')
    model = BiEncoder(config)
    tokenizer = BertTokenizer.from_pretrained(config.bert_path)
    print('load the pretrained model done...')
    if config.parall and config.device is 'cuda':
        if torch.cuda.device_count() > 1:
            print('Use', torch.cuda.device_count(), 'gpus')
            model = nn.DataParallel(model)
    model.to(device)
    print('fine tune the model...')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.lr)
    log = {}
    for epoch in range(config.n_epoch):
        random.shuffle(data)
        n = 0
        batch_size = config.batch_size
        data_num = len(data)
        n_batch = 0
        while n < data_num - 1 and n_batch < config.max_batches:
            optimizer.zero_grad()
            n_batch += 1
            batch_data = data[n:n + batch_size]
            label_batch = [i[0] for i in batch_data]
            name_batch = [i[1] for i in batch_data]
            synonym_batch = [i[2] for i in batch_data]
            linking_label_batch = [i[3] for i in batch_data]
            linking_label_batch = Tensor(linking_label_batch, device=device)
            # print(linking_label_batch)
            # print(name_batch)
            # print(synonym_batch)
            entity_input = tokenizer.batch_encode_plus(name_batch, padding=True, max_length=128)
            entity_input_ids = Tensor(entity_input['input_ids'], device=device)
            entity_attention_mask = Tensor(entity_input['attention_mask'], device=device)
            entity_token_type_ids = Tensor(entity_input['token_type_ids'], device=device)
            mention_input = tokenizer.batch_encode_plus(synonym_batch, padding=True, max_length=128)
            mention_input_ids = Tensor(mention_input['input_ids'], device=device)
            mention_attention_mask = Tensor(mention_input['attention_mask'], device=device)
            mention_token_type_ids = Tensor(mention_input['token_type_ids'], device=device)
            entity_embedding, mention_embedding, loss = model(entity_input_ids,entity_attention_mask,entity_token_type_ids,
                  mention_input_ids,mention_attention_mask,mention_token_type_ids,
                  linking_label_batch)
            n += batch_size
            optimizer.zero_grad()
            # print("loss: ",loss)
            loss = loss.sum()
            # print("loss: ", loss)
            if log.get(epoch+1) is None:
                log[epoch+1] = []
            log[epoch+1].append({"batch:loss":str(n_batch)+':'+str(loss.item())})
            if n_batch % 8 == 0:
                print("*"*15, " epoch: {}, batch: {} ".format(epoch+1,n_batch), "*"*15)
                print("loss: {}".format(loss.item()))
                with open('./TrainBiencoderLog.json','w',encoding='utf-8') as f:
                    f.write(json.dumps(log))
            loss.backward()
            optimizer.step()
        if (epoch + 1) % config.save_step == 0:
            print('save the model...')
            if not os.path.exists(config.model_save_path):
                os.mkdir(config.model_save_path)
            torch.save(model.state_dict(), config.model_save_path + '/' + 'model' + '_' + str(epoch + 1) + '.bin')
            print('save the model done...')

    print('fine tune the model done...')


if __name__ == '__main__':
    config = ConfigBiEncoder()
    train(config)
