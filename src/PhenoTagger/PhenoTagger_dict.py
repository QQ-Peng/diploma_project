# Date: 2021/4/11
# Author: Qianqian Peng
# Reference: https://github.com/ncbi-nlp/PhenoTagger/blob/master/src/PhenoTagger_tagging.py

import argparse
from nn_model import bioTag_BERT
from dic_ner import dic_ont
from tagging_text import bioTag, bioTag_dic
import os
import time
import json
import re


def PubTator_Converter(infile, outfile, biotag_dic, para_set):
	with open(infile, 'r', encoding='utf-8') as fin:
		with open(outfile, 'w', encoding='utf8') as fout:
			title = ''
			abstract = ''
			for line in fin:
				# print('*'*20)
				# print('read file')
				# print(line)
				line = line.rstrip()
				p_title = re.compile('^([0-9]+)\|t\|(.*)$')
				p_abstract = re.compile('^([0-9]+)\|a\|(.*)$')
				if p_title.search(line):  # title
					m = p_title.match(line)
					pmid = m.group(1)
					title = m.group(2)
					fout.write(pmid + "|t|" + title + "\n")
				elif p_abstract.search(line):  # abstract
					m = p_abstract.match(line)
					pmid = m.group(1)
					abstract = m.group(2)
					fout.write(pmid + "|a|" + abstract + "\n")
				# fixme: abstract may be empty!
				if abstract != '':  # annotation
					# print('annotate file')
					intext = title + ' ' + abstract
					# print(intext)
					tag_result = bioTag_dic(intext, biotag_dic, onlyLongest=False, abbrRecog=False)
					# print(tag_result)
					for ele in tag_result:
						start = ele[0]
						last = ele[1]
						mention = intext[int(ele[0]):int(ele[1])]
						type = 'MeSH'
						id = ele[2]
						score = ele[3]
						# print('write outfile')
						fout.write(
							pmid + "\t" + start + "\t" + last + "\t" + mention + "\t" + type + "\t" + id + "\t" + score + "\n")
					fout.write('\n')
					title = ''
					abstract = ''

def phenotagger_tag(infolder, para_set, outfolder):
	ontfiles = {'dic_file': '../dict/noabb_lemma.dic',
	            'word_hpo_file': '../dict/word_id_map.json',
	            'hpo_word_file': '../dict/id_word_map.json'}


	vocabfiles = {'labelfile': '../dict/lable.vocab',
		          'config_path': '../models/biobert_v11_pubmed/bert_config.json',
		          'checkpoint_path': '../models/biobert_v11_pubmed/model.ckpt-1000000',
		          'vocab_path': '../models/biobert_v11_pubmed/vocab.txt'}

	# loading dict and model

	biotag_dic = dic_ont(ontfiles)

	# tagging text
	print("begin tagging........")
	start_time = time.time()

	i = 0
	N = 0
	for filename in os.listdir(infolder):
		N += 1
	for filename in os.listdir(infolder):
		print("Processing:{0}%".format(round(i * 100 / N)), end="\r")
		i += 1

		with open(infolder + filename, 'r', encoding='utf-8') as fin:
			format = ""
			for line in fin:
				pattern_bioc = re.compile('.*<collection>.*')
				pattern_pubtator = re.compile('^([^\|]+)\|[^\|]+\|(.*)')
				format = "PubTator"

			PubTator_Converter(infolder + filename, outfolder + filename, biotag_dic, para_set)

	print('tag done:', time.time() - start_time)


if __name__ == "__main__":

	parser = argparse.ArgumentParser(
		description='build weak training corpus, python build_dict.py -i infile -o outpath')
	parser.add_argument('--infolder', '-i', help="input folder path", default='../example/input/')
	parser.add_argument('--outfolder', '-o', help="output folder path", default='../example/output/')

	args = parser.parse_args()
	args.outfolder = args.outfolder + '/'
	args.infolder = args.infolder + '/'
	if not os.path.exists(args.outfolder):
		os.makedirs(args.outfolder)

	para_set = {
		'onlyLongest': False,  # False: return overlap concepts, True only longgest
		'abbrRecog': True,  # False: don't identify abbr, True: identify abbr
	}

	phenotagger_tag(args.infolder, para_set, args.outfolder)
	print('The results are in ', args.outfolder)