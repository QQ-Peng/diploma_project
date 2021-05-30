# Date: 2021-4-12
# Author: Qianqian Peng

import re
import sys, os
import argparse
import json

def extract_MeSH_and_Mention(args):
	infolder = args.infolder + '/'
	outfolder = args.outfolder + '/'
	all_MeSH_term = []
	all_Mention = []
	all_pmid = []
	file_list = os.listdir(infolder)
	n = 0
	for file in file_list:
		n += 1
		print('Processing {}: {}/{}'.format(file, n, len(file_list)))
		pmid = file.split('.')[0][4:]
		print(pmid)
		all_pmid.append(pmid)
		sentence_pattern = re.compile(r'Processing {}.PubTator.tx.[0-9]*: '.format(pmid))
		phrase_pattern = re.compile(r'Phrase: ')
		map_pattern = re.compile(r'Meta Mapping \([0-9]*\):')
		annotation_pattern = re.compile(r'\(.*{')
		data = open(infolder+file,'r',encoding='utf-8').read()
		split_data = sentence_pattern.split(data)
		length = len(split_data)

		# split the data
		for _ in range(length):
			elem = split_data.pop(0)
			if elem != '':
				elem = elem.strip()
				split_phrase = phrase_pattern.split(elem)
				meta_mapping = []
				length2 = len(split_phrase)
				for _ in range(length2):
					elem2 = split_phrase.pop(0)
					elem2 = elem2.strip()
					split_meta = map_pattern.split(elem2)
					if len(split_meta) > 1:
						meta_mapping.append(split_meta)
				split_data.append(meta_mapping)
		# strip every line
		for sentence_i in range(len(split_data)):
			for phrase_i in range(len(split_data[sentence_i])):
				for mapping_i in range(len(split_data[sentence_i][phrase_i])):
					split_data[sentence_i][phrase_i][mapping_i] = split_data[sentence_i][phrase_i][mapping_i].strip()

		complete_pattern = re.compile(r'\(.*}\)') # egg: 688   SPECTROPHOTOMETRY (Spectrophotometry {AOD,CHV,CSP,HL7V3.0,LCH,LCH_NW,LNC,MSH,MTH,NCI,NCI_CDISC,NCI_CTRP,PDQ,SNMI,SNOMEDCT_US}) [Laboratory Procedure]
		incomplete_pattern = re.compile(r' {.*}.*') # egg: 623   Flavodoxin {CSP,MSH,MTH,SNMI,SNOMEDCT_US} [Amino Acid, Peptide, or Protein,Biologically Active Substance]
		pattern_for_extract_mention_in_complete_pattern = re.compile(r' \(.*{.*}\).*')
		MeSH_term = []
		Mention = []
		for sentence in split_data:
			for phrase in sentence:
				for mapping in phrase:
					token = mapping.split('\n')
					for token_one in token:
						token_one = token_one.strip()
						if 'MSH' in token_one:
							if len(complete_pattern.findall(token_one)) > 0:
								token_one_split = annotation_pattern.split(token_one)
								msh = token_one[len(token_one_split[0]) + 1:token_one.index(token_one_split[1]) - 1].strip()
								mention = pattern_for_extract_mention_in_complete_pattern.split(token_one)[0].\
									split(' ')[1:]
								mention = ' '.join(mention).strip()
								MeSH_term.append(msh)
								Mention.append(mention)
							else:
								token_one_split = incomplete_pattern.split(token_one)[0].split(' ')[1:]
								mention = msh = ' '.join(token_one_split).strip()
								MeSH_term.append(msh)
								Mention.append(mention)
		all_MeSH_term.append(MeSH_term)
		all_Mention.append(Mention)
	outfile = outfolder + 'abstract_to_MeSH.jsonl'
	outfile = open(outfile,'w',encoding='utf-8',newline='\n')
	for i in range(len(all_pmid)):
		new_record = {}
		new_record['pmid'] = all_pmid[i]
		new_record['meshMajor'] = list(set(all_MeSH_term[i]))
		new_record['mention'] = list(set(all_Mention[i]))
		outfile.write(json.dumps(new_record)+'\n')
	outfile.close()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description='Extract MeSH term from MetaMap annotation.')
	parser.add_argument('--infolder', '-i', help="input folder path", default='../data/test_data/annotation_metamap/')
	parser.add_argument('--outfolder', '-o', help="output folder path", default='../data/result/')
	args = parser.parse_args()
	if not os.path.exists(args.infolder):
		out_str = '{} does not exists.'.format(args.infolder)
		raise ValueError(out_str)
	if not os.path.exists(args.outfolder):
		os.mkdir(args.outfolder)
	extract_MeSH_and_Mention(args)