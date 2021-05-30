# Date: 
# Author:

import torch
from transformers import BertModel,BertTokenizer
from config import *
class aaa(torch.nn.Module):
	def __init__(self):
		super(aaa,self).__init__()
		self.layer=torch.nn.Linear(100,100)
	def forward(self,x):
		return self.layer(x)

# model = aaa()
# model.to('cuda')
# x=torch.randn(1,100).to('cuda')
# print(x)

config = ConfigBiEncoder()
# config.bert_path = '/home/pqq/pqq/PyProject/mention_detection/model/bert-base-uncased'
device = 'cpu'
bert_model= BertModel.from_pretrained(config.bert_path)
bert_model.to(device)

tokenizer = BertTokenizer.from_pretrained(config.bert_path)
input = tokenizer.encode('i love china, what about you? you love usa')

input1=torch.tensor([input],device=device)
# input2=torch.tensor([input]).to('cuda')
#
print("input1 output: ")
print(bert_model(input1)[1])
# print("input2 output: ")
# print(bert_model(input2))

