# Date: 2021-5-7
# Author: Qianqian Peng
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
import os

from text_processor import sent_tokenize, word_tokenize, POS_tag
from utlis import convert_obo_dict, generate_candidate

n = 0


def tag_dict(ontology: dict, sentence_tokenizer, token_tokenzier, file, threshold=0.8):
	global n
	n += 1
	text = open(file, 'r', encoding='utf-8').read()
	sentences = sent_tokenize(sentence_tokenizer, text)
	sent_num = len(sentences)
	for _ in range(sent_num):
		sent = sentences.pop(0)
		sent = sent if '|t|' not in sent and '|a|' not in sent else \
			sent.split('|t|')[1] if '|t|' in sent else sent.split('|a|')[1]
		sentences.append(sent)
	tokens = word_tokenize(token_tokenzier, sentences)
	detokenizer = TreebankWordDetokenizer().detokenize
	candidates = [generate_candidate(sent,detokenizer) for sent in tokens]
	count = 0
	for i in candidates:
		for j in i:
			count += 1
	# print(candidates)
	mesh = []
	match_n = 0
	for sent in candidates:
		for token in sent:
			if ontology.get(token) is not None:
				mesh += token
			else:
				for name in ontology.keys():
					if name[0] == '"' and name[-1] == '"':
						name = name[1:-1]
					dist = nltk.edit_distance(token, name)
					if dist / len(name) < (1 - threshold) or dist / len(token) < (1 - threshold):
						mesh.append(name)
					# print("token: {}, name: {}".format(token,name))
			match_n += 1
			if match_n % 10 == 0:
				print("article: {}, process candidat: {}/{}".format(n,match_n,count))

	# print(mesh)


def main():
	ontology_path = '../ontology/mesh_addsyn.obo'
	id2name, name2id = convert_obo_dict(ontology_path)
	file_dir = '../data/test_data/Articles_PubTator/'
	sentence_tokenizer = nltk.sent_tokenize
	token_tokenizer = nltk.word_tokenize
	# print(type(name2id))
	for file in os.listdir(file_dir):
		tag_dict(name2id, sentence_tokenizer, token_tokenizer, file_dir + file, threshold=0.8)


if __name__ == '__main__':
	main()
