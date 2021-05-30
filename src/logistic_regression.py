# Date: 2021-5-24
# Author: Qianqian Peng

import json
import random
import torch
import torch.nn as nn
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn import feature_selection as FS
from sklearn import linear_model,svm
import numpy as np
from joblib import dump,load
from sklearn.feature_selection import SelectKBest
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

def check_label(data_path='F:/Data/allMeSH_2020/allMeSH_2020.json',
                dict_path='../dict/all_name_id.txt',
                out='../data/case_regression/mesh_covered.txt'):
	name2id_data = open(dict_path, 'r', encoding='utf-8').read().split('\n')
	if name2id_data[-1] == '':
		name2id_data.pop(-1)
	name2id = {}
	for line in name2id_data:
		line = line.split('::')
		name2id[line[0]] = line[1]
	covered_label = []
	with open(data_path, 'r', encoding='iso8859') as f:
		n = 0
		for line in f:
			line = line.strip()
			if line[-1] == ',':
				line = line[:-1]
				n += 1
			elif line[-1] == '}':
				line = line[:-2]
				n += 1
			else:
				continue

			record = json.loads(line)

			mesh = record['meshMajor']
			for m in mesh:
				if name2id.get(m) is not None:
					covered_label.append(m)
			if n % 10000 == 0:
				print("process {} articles.".format(n))
		covered_label = list(set(covered_label))
		print("covered label number: {}".format(len(covered_label)))
		out_line = '\n'.join(covered_label)
		out_file = open(out, 'w', encoding='utf-8', newline='\n')
		out_file.write(out_line)
		out_file.close()


def split_trainData_label(data_path):
	label2pmid = {}
	with open(data_path, 'r', encoding='iso8859') as f:
		n = 0
		for line in f:
			line = line.strip()
			if line[-1] == ',':
				line = line[:-1]
				n += 1
			elif line[-1] == '}':
				line = line[:-2]
				n += 1
			else:
				continue
			record = json.loads(line)
			pmid = record['pmid']
			mesh = record['meshMajor']
			for one_label in mesh:
				if label2pmid.get(one_label) is None:
					label2pmid[one_label] = [pmid]
				else:
					label2pmid[one_label].append(pmid)
			if n % 10000 == 0:
				print('process {} articles.'.format(n))
				with open('../data/label2pmid/' + '/label2pmid_' + str(n) + '.json', 'w', newline='\n') as out:
					out.write(json.dumps(label2pmid))


def build_trainData(data_path, label2pmid, target_label='Kidney Glomerulus',
                    max_train=100, out_path='../data/data_ML/'):
	target_articles_num = len(label2pmid[target_label])
	positive_pmid = label2pmid[target_label]
	if max_train > target_articles_num:
		max_train = int(target_articles_num*2/3)
	for_search_dict = {pmid: 0 for pmid in positive_pmid}
	# out_path = out_path + '/' + target_label + '/'
	if not os.path.exists(out_path):
		os.mkdir(out_path)
	pos_pmid = open(out_path + '/' + 'pos_pmid.txt', 'w', encoding='utf-8', newline='\n')
	neg_pmid = open(out_path + '/' + 'neg_pmid.txt', 'w', encoding='utf-8', newline='\n')
	with open(data_path, 'r', encoding='iso8859') as f:
		positive_out = open(out_path + '/positive_data.jsonl', 'w', encoding='iso8859', newline='\n')
		negative_out = open(out_path + '/negative_data.jsonl', 'w', encoding='iso8859', newline='\n')
		n = 0
		num_neg = 0
		num_pos = 0
		for line in f:
			line = line.strip()
			if line[-1] == ',':
				line = line[:-1]
				n += 1
			elif line[-1] == '}':
				line = line[:-2]
				n += 1
			else:
				continue
			record = json.loads(line)
			pmid = record['pmid']
			if for_search_dict.get(pmid) is not None:
				positive_out.write(line + '\n')
				pos_pmid.write(pmid + '\n')
				num_pos += 1
				if num_pos == max_train or num_pos == target_articles_num:
					break
			else:
				if num_neg < max_train:
					select_p = random.random()
					if select_p < 0.1:
						negative_out.write(line + '\n')
						neg_pmid.write(pmid + '\n')
						num_neg += 1
		positive_out.close()
		negative_out.close()
		pos_pmid.close()
		neg_pmid.close()


def combine_pos_neg(path):
	out = path + '/data.txt'
	out = open(out, 'w', encoding='utf-8', newline='\n')
	with open(path + '/positive_data.jsonl', 'r', encoding='iso8859') as f:
		for line in f:
			line = json.loads(line)
			abstract = line['abstractText']
			title = line['title']
			if title is not None:
				out.write(title + ' ' + abstract + '\n')
			else:
				out.write(abstract + '\n')

	with open(path + '/negative_data.jsonl', 'r', encoding='iso8859') as f:
		for line in f:
			line = json.loads(line)
			abstract = line['abstractText']
			title = line['title']
			if title is not None:
				out.write(title + ' ' + abstract + '\n')
			else:
				out.write(abstract + '\n')

	out.close()


def build_testData(data_path,
                   label2pmid: dict,
                   pos_pmid_train_path,
                   neg_pmid_train_path,
                   target_label='Kidney Glomerulus',
                   max_num=100,
                   out_path='./'):
	pos_pmid_train = open(pos_pmid_train_path, 'r').read().split('\n')
	neg_pmid_train = open(neg_pmid_train_path, 'r').read().split('\n')
	target_pmid = label2pmid[target_label]
	if max_num > len(target_pmid):
		max_num = int(len(target_pmid)/3)
	target_pmid_dict = {pm: 1 for pm in target_pmid}
	pos_train_dict = {pm: 1 for pm in pos_pmid_train}
	neg_train_dict = {pm: 1 for pm in neg_pmid_train}

	if pos_pmid_train[-1] == '':
		pos_pmid_train.pop(-1)
	if neg_pmid_train[-1] == '':
		neg_pmid_train.pop(-1)

	with open(data_path, 'r', encoding='iso8859') as f:
		positive_out = open(out_path + '/positive_data_test.jsonl', 'w', encoding='iso8859', newline='\n')
		negative_out = open(out_path + '/negative_data_test.jsonl', 'w', encoding='iso8859', newline='\n')
		n = 0
		num_neg = 0
		num_pos = 0
		for line in f:
			if n % 10000==0:
				print(n)
			line = line.strip()
			if line[-1] == ',':
				line = line[:-1]
				n += 1
			elif line[-1] == '}':
				line = line[:-2]
				n += 1
			else:
				continue
			record = json.loads(line)
			pmid = record['pmid']
			if target_pmid_dict.get(pmid) is not None and pos_train_dict.get(pmid) is None:
				positive_out.write(line + '\n')
				num_pos += 1
				if num_pos == max_num or num_pos == len(target_pmid):
					break
			else:
				if num_neg < max_num and target_pmid_dict.get(pmid) is None and \
						neg_train_dict.get(pmid) is None:
					select_p = random.random()
					if select_p < 0.05:
						negative_out.write(line + '\n')
						num_neg += 1
		positive_out.close()
		negative_out.close()


def split_text(data_path, out_path):
	lemmatizer = WordNetLemmatizer()
	data = open(data_path, 'r', encoding='utf-8').read().strip()
	words = nltk.word_tokenize(data)
	words = [w.lower() for w in words]
	stop_words = list(set(stopwords.words('english')))
	stop_words.append('')
	words = [lemmatizer.lemmatize(w) for w in words if not w in stop_words]
	words = list(set(words))
	with open(out_path, 'w', encoding='utf-8', newline='\n') as f:
		out_line = '\n'.join(words)
		f.write(out_line)


def feature_selection(words_path, positive_path, negtive_path):
	lemmatizer = WordNetLemmatizer()
	all_words = open(words_path, 'r', encoding='utf-8').read().split('\n')
	if all_words[-1] == '':
		all_words.pop(-1)
	word2idx = {}
	for i in range(len(all_words)):
		word2idx[all_words[i]] = i

	stop_words = list(set(stopwords.words('english')))
	stop_words.append('')
	pos_text = open(positive_path, 'r', encoding='iso8859').read().split('\n')
	neg_text = open(negtive_path, 'r', encoding='iso8859').read().split('\n')
	if pos_text[-1] == '':
		pos_text.pop(-1)
	if neg_text[-1] == '':
		neg_text.pop(-1)
	pos_feature = []
	neg_feature = []
	for i in range(len(pos_text)):
		one_pos_record = json.loads(pos_text[i])
		if one_pos_record['title'] is None:
			one_pos_record['title'] = ' '
		one_pos_words = nltk.word_tokenize(one_pos_record['title'] + one_pos_record['abstractText'])
		one_pos_words = [w.lower() for w in one_pos_words]
		one_pos_words = [lemmatizer.lemmatize(w) for w in one_pos_words if not w in stop_words]
		word2count = {}
		for w in one_pos_words:
			if word2count.get(w) is None:
				word2count[w] = one_pos_words.count(w)
		feature_vec = []
		for w in word2idx.keys():
			if word2count.get(w) is None:
				feature_vec.append(0)
			else:
				feature_vec.append(word2count[w])
		pos_feature.append(feature_vec)

	for i in range(len(neg_text)):
		one_neg_record = json.loads(neg_text[i])
		if one_neg_record['title'] is None:
			one_neg_record['title'] = ' '
		one_neg_words = nltk.word_tokenize(one_neg_record['title'] + one_neg_record['abstractText'])
		one_neg_words = [w.lower() for w in one_neg_words]
		one_neg_words = [lemmatizer.lemmatize(w) for w in one_neg_words if not w in stop_words]
		word2count = {}
		for w in one_neg_words:
			if word2count.get(w) is None:
				word2count[w] = one_neg_words.count(w)
		feature_vec = []
		for w in word2idx.keys():
			if word2count.get(w) is None:
				feature_vec.append(0)
			else:
				feature_vec.append(word2count[w])
		neg_feature.append(feature_vec)
	pos_label = [1] * len(pos_feature)
	neg_label = [0] * len(neg_feature)

	skb = SelectKBest(score_func=FS.f_classif,k=int(len(pos_feature[0])*0.01))
	new_x = skb.fit_transform(pos_feature + neg_feature, pos_label + neg_label)
	select_idx = skb.get_support()
	selected_words = []
	for i in range(len(select_idx)):
		if select_idx[i] == True:
			selected_words.append(all_words[i])

	return all_words,selected_words

def feature_selection2(words_path, positive_path, negtive_path):
	train_pos_text = open(positive_path, 'r', encoding='iso8859').read().split('\n')
	train_neg_text = open(negtive_path, 'r', encoding='iso8859').read().split('\n')
	pop_tail(train_pos_text, '')
	pop_tail(train_neg_text, '')
	train_text = train_pos_text + train_neg_text
	train_label = [1] * len(train_pos_text) + [0] * len(train_neg_text)
	vectorizer = CountVectorizer() # count word frequent
	transformer = TfidfTransformer() # count tfidf
	X = vectorizer.fit_transform(train_text)
	words_freq = X.toarray()
	all_words = vectorizer.get_feature_names()
	tfidf = transformer.fit_transform(X)
	tfidf_weight = tfidf.toarray()
	# x_train = (torch.tensor(words_freq)*torch.tensor(tfidf_weight)).tolist()
	x_train = torch.tensor(words_freq)
	# x_train = (x_train/x_train.sum(dim=-1).unsqueeze(-1)).tolist()
	x_train = x_train.tolist()
	y_train = train_label
	skb = SelectKBest(score_func=FS.f_classif,k=int(len(all_words)*0.01))
	new_x = skb.fit_transform(x_train, y_train)
	select_idx_mask = skb.get_support()
	select_idx = []
	for i in range(len(select_idx_mask)):
		if select_idx_mask[i] == True:
			select_idx.append(i)
	selected_words = []
	for i in select_idx:
		selected_words.append(all_words[i])

	return all_words,selected_words,select_idx


def load_data(pos_path, neg_path, feature_words_path, max_num=400):
	lemmatizer = WordNetLemmatizer()
	feature_words = open(feature_words_path, 'r', encoding='iso8859').read().split('\n')

	stop_words = list(set(stopwords.words('english')))
	stop_words.append('')
	if feature_words[-1] == '':
		feature_words.pop(-1)
	pos_data = []
	neg_data = []
	with open(pos_path, 'r', encoding='iso8859') as f:
		n = 0
		for line in f:
			line = json.loads(line)
			if line['title'] is None:
				line['title'] = ' '
			text = line['title'] + ' ' + line['abstractText']
			text = nltk.word_tokenize(text.lower())
			text = [lemmatizer.lemmatize(w) for w in text if not w in stop_words]
			feature_vec = [text.count(w) for w in feature_words]
			pos_data.append(feature_vec)
			n += 1
			if n == max_num:
				break
	with open(neg_path, 'r', encoding='iso8859') as f:
		n = 0
		for line in f:
			line = json.loads(line)
			if line['title'] is None:
				line['title'] = ' '
			text = line['title'] + ' ' + line['abstractText']
			text = nltk.word_tokenize(text.lower())
			text = [lemmatizer.lemmatize(w) for w in text if not w in stop_words]
			feature_vec = [text.count(w) for w in feature_words]
			neg_data.append(feature_vec)
			n += 1
			if n == max_num:
				break
	train_data = pos_data + neg_data
	train_label = [1] * len(pos_data) + [0] * len(neg_data)
	return train_data, train_label


class LogisticRegression(nn.Module):
	def __init__(self, dim):
		super(LogisticRegression, self).__init__()
		self.lr = nn.Linear(dim, 1)
		# self.lr1 = nn.Linear(222, 1)
		self.sm = nn.Sigmoid()

	def forward(self, x):
		x = self.lr(x)
		# x = self.lr1(x)
		x = self.sm(x)
		return x


class SVM(nn.Module):
	def __init__(self, dim):
		super(SVM, self).__init__()
		self.lr = nn.Linear(dim, 1)

	def forward(self, x):
		x = self.lr(x)
		# x = self.lr1(x)
		return x

	def compute_loss(self, output, label):
		loss = torch.mean(torch.clamp(1 - output.t() * label, min=0))  # hinge loss
		loss += 0.01 * torch.mean(self.lr.weight ** 2)  # l2 penalty
		return loss


def logiatic_regress_train(pos_path, neg_path,
                           pos_path_test, neg_path_test,
                           feature_words_path):
	train_data, train_label = load_data(pos_path, neg_path, feature_words_path)
	test_data, test_label = load_data(pos_path_test, neg_path_test, feature_words_path)
	print('build training data done.')
	logistic_model = LogisticRegression(len(train_data[0]))
	svm_model = SVM(len(train_data[0]))
	if torch.cuda.is_available():
		logistic_model.to('cpu')
		svm_model.to('cpu')

	# 定义损失函数和优化器
	criterion = nn.BCELoss()
	optimizer = torch.optim.SGD(logistic_model.parameters(), lr=1e-5, momentum=0.9)
	optimizer_svm = torch.optim.SGD(svm_model.parameters(), lr=1e-5, momentum=0.9)
	threshold = 0.5
	logreg = linear_model.LogisticRegression(C=1e3,solver='liblinear')
	logreg.fit(train_data, train_label)
	svm_sk = svm.SVC()
	svm_sk.fit(train_data,train_label)
	for _ in range(100):
		rand_idx = list(range(len(train_data)))
		random.shuffle(rand_idx)
		train_data_rand = [train_data[i] for i in rand_idx]
		train_label_rand = [train_label[i] for i in rand_idx]
		step = 50
		n = 0
		acc_all = []
		acc_all_svm = []
		while n < len(train_data_rand):
			x_data = train_data_rand[n:n + step]
			y_data = train_label_rand[n:n + step]
			x_data = torch.tensor(x_data).float().to('cpu')
			y_data = torch.tensor(y_data).float().unsqueeze(-1).to('cpu')
			out = logistic_model(x_data)
			loss = criterion(out, y_data)
			out_svm = svm_model(x_data)
			loss_svm = svm_model.compute_loss(out_svm, y_data)
			print_loss = loss.data.item()
			# print('loss: ',print_loss)
			mask = out.ge(threshold).float()  # 以0.5为阈值进行分类
			mask_svm = out_svm.ge(threshold).float()  # 以0.5为阈值进行分类
			correct = (mask == y_data).sum()  # 计算正确预测的样本个数
			correct_svm = (mask_svm == y_data).sum()  # 计算正确预测的样本个数
			acc = correct.item() / x_data.size(0)  # 计算精度
			acc_svm = correct_svm.item() / x_data.size(0)  # 计算精度
			optimizer.zero_grad()
			optimizer_svm.zero_grad()
			loss.backward()
			loss_svm.backward()
			optimizer.step()
			optimizer_svm.step()

			# print(acc)
			acc_all.append(acc)
			acc_all_svm.append(acc_svm)
			n += step

		print('*' * 20)
		print('train overall acc: ', sum(acc_all) / len(acc_all))
		print('train overall acc_svm: ', sum(acc_all_svm) / len(acc_all_svm))
		# for test model
		with torch.no_grad():
			x_data = torch.tensor(test_data).float().to('cpu')
			y_data = torch.tensor(test_label).float().unsqueeze(-1).to('cpu')
			out = logistic_model(x_data)
			# print(out)
			out_svm = svm_model(x_data)
			mask = out.ge(threshold).float()  # 以0.5为阈值进行分类
			mask_svm = out_svm.ge(threshold).float()  # 以0.5为阈值进行分类
			correct = (mask == y_data).sum()  # 计算正确预测的样本个数
			correct_svm = (mask_svm == y_data).sum()  # 计算正确预测的样本个数
			acc = correct.item() / x_data.size(0)
			acc_svm = correct_svm.item() / x_data.size(0)

			acc_sk = logreg.score(test_data,test_label)
			acc_svm_sk = svm_sk.score(test_data,test_label)
			print("test acc: ", acc)
			print("test acc_svm: ", acc_svm)
			print("test acc sk: ",acc_sk)
			print("test acc svm sk: ",acc_svm_sk)
	dump(logreg,'../data/case_regression/logreg.joblib')
	dump(svm_sk,'../data/case_regression/svm.joblib')

def pop_tail(vec,tail_c=''):
	if vec[-1] == tail_c:
		vec.pop(-1)

def logiatic_regress_train2(pos_path, neg_path,
                           pos_path_test, neg_path_test,
                           feature_words_path):
	train_pos_text = open(pos_path,'r',encoding='iso8859').read().split('\n')
	train_neg_text = open(neg_path,'r',encoding='iso8859').read().split('\n')
	pop_tail(train_pos_text,'')
	pop_tail(train_neg_text,'')
	train_text = train_pos_text + train_neg_text
	train_label = [1]*len(train_pos_text)+[0]*len(train_neg_text)
	vectorzier = CountVectorizer() # count word frequent
	X = vectorzier.fit_transform(train_text)
	words_freq = X.toarray()

	transformer = TfidfTransformer()
	tfidf = transformer.fit_transform(X)
	tfidf_weight = tfidf.toarray()
	# all_feature_vec = torch.tensor(words_freq)*torch.tensor(tfidf_weight)
	all_feature_vec = torch.tensor(words_freq)
	#all_feature_vec = all_feature_vec/all_feature_vec.sum(dim=-1).unsqueeze(-1)
	all_feature_vec = all_feature_vec
	current_words = vectorzier.get_feature_names()
	feature_words = open(feature_words_path, 'r', encoding='iso8859').read().split('\n')
	pop_tail(feature_words, '')
	feature_words_idx = []
	for w in feature_words:
		feature_words_idx.append(current_words.index(w))

	selected_feature_vec = []
	for i in feature_words_idx:
		selected_feature_vec.append(all_feature_vec[:,i].unsqueeze(-1))
	selected_feature_vec = torch.cat(selected_feature_vec,dim=-1).tolist()
	#train model
	logreg = linear_model.LogisticRegression(C=1e4, solver='liblinear')
	logreg.fit(selected_feature_vec, train_label)
	svm_sk = svm.SVC()
	svm_sk.fit(selected_feature_vec, train_label)
	# test model
	test_pos_text = open(pos_path_test, 'r', encoding='iso8859').read().split('\n')
	test_neg_text = open(neg_path_test, 'r', encoding='iso8859').read().split('\n')
	pop_tail(test_pos_text, '')
	pop_tail(test_neg_text, '')
	test_text = test_pos_text + test_neg_text
	test_label = [1] * len(test_pos_text) + [0] * len(test_neg_text)
	vectorzier_test = CountVectorizer()  # count word frequent
	X_test = vectorzier_test.fit_transform(test_text)
	current_words_test = vectorzier_test.get_feature_names()
	feature_words_idx_test = []
	for w in feature_words:
		if w in current_words_test:
			feature_words_idx_test.append(current_words_test.index(w))
		else:
			feature_words_idx_test.append(-1)


	words_freq = torch.tensor(X_test.toarray())
	# words_freq = words_freq/words_freq.sum(dim=-1).unsqueeze(-1)
	selected_feature_vec = []
	for i in feature_words_idx_test:
		if i == -1:
			selected_feature_vec.append(torch.tensor([0]*len(test_label)).unsqueeze(-1))
		else:
			selected_feature_vec.append(words_freq[:,i].unsqueeze(-1))
	selected_feature_vec = torch.cat(selected_feature_vec,dim=-1).tolist()
	acc_sk = logreg.score(selected_feature_vec, test_label)
	acc_svm_sk = svm_sk.score(selected_feature_vec, test_label)
	print("test logistic regression acc. : ",acc_sk)
	print("test svm acc. : ",acc_svm_sk)
	dump(logreg,'../data/case_regression/logreg2.joblib')
	dump(svm_sk,'../data/case_regression/svm.2joblib')



# with open('../data/case_regression/positive_data_test.jsonl','r',encoding='iso8859') as f:
# 	out = open('../data/case_regression/positive_data_test.txt','w',encoding='utf-8',newline='\n')
# 	for line in f:
# 		line=json.loads(line)
# 		title = line['title']
# 		if line['title'] is None:
# 			title = ' '
# 		out.write(title+line['abstractText']+'\n')
# 	out.close()
#
# with open('../data/case_regression/negative_data_test.jsonl','r',encoding='iso8859') as f:
# 	out = open('../data/case_regression/negative_data_test.txt','w',encoding='utf-8',newline='\n')
# 	for line in f:
# 		line=json.loads(line)
# 		title = line['title']
# 		if line['title'] is None:
# 			title = ' '
# 		out.write(title+line['abstractText']+'\n')
# 	out.close()
# import sys
# sys.exit()

print("load label2pmid")
label2pmid = json.loads(open('../data/label2pmid_sample/label2pmid_1000000.json','r').read())
print("load label2pmid done")
print('start select training data.')
build_trainData('/media/dragon/F/Data/allMeSH_2020/allMeSH_2020.json',label2pmid,
				'Nuclear Warfare',100,'../data/case_regression/')
print('select training data done.')
combine_pos_neg('../data/case_regression/')
print('split the text to words.')
split_text('../data/case_regression/data.txt', '../data/case_regression/words.txt')
print('split the text to words done.')
print('start select feature.')
words,selected_words = feature_selection('../data/case_regression/words.txt','../data/case_regression/positive_data.jsonl',
								 '../data/case_regression/negative_data.jsonl')

# words,selected_words, selected_idx = feature_selection2('../data/case_regression/words.txt','../data/case_regression/positive_data_train.txt',
# 								 '../data/case_regression/negative_data_train.txt')


# index = []
# for i in range(len(pv)):
# 	if 0.001<=pv[i]<=0.05:
# 		index.append(i)
out_feature_words = open('../data/case_regression/feature_words.txt','w',encoding='iso8859',newline='\n')
# out_feature_words = open('../data/case_regression/feature_words2.txt','w',encoding='iso8859',newline='\n')
#
# for i in index:
# 	out_feature_words.write(words[i]+'\n')
out_feature_words.write('\n'.join(selected_words))
out_feature_words.close()
print('select feature done.')
build_testData('/media/dragon/F/Data/allMeSH_2020/allMeSH_2020.json',
			   label2pmid,'../data/case_regression/pos_pmid.txt',
			   '../data/case_regression/neg_pmid.txt',
			   'Nuclear Warfare',200,'../data/case_regression/')
print('start train model')
logiatic_regress_train('../data/case_regression/positive_data.jsonl',
                       '../data/case_regression/negative_data.jsonl',
                       '../data/case_regression/positive_data_test.jsonl',
                       '../data/case_regression/negative_data_test.jsonl',
                       '../data/case_regression/feature_words.txt')
#
# logiatic_regress_train2('../data/case_regression/positive_data_train.txt',
#                        '../data/case_regression/negative_data_train.txt',
#                        '../data/case_regression/positive_data_test.txt',
#                        '../data/case_regression/negative_data_test.txt',
#                        '../data/case_regression/feature_words2.txt')
# print('start train model done.')
if __name__ == '__main__':
	# split_trainData_label('F:/Data/allMeSH_2020/allMeSH_2020.json')
	# build_trainData('F:/Data/allMeSH_2020/allMeSH_2020.json')
	# split_text('../data/case_regression/data.txt','../data/case_regression/words.txt')
	# pv, fv = feature_selection('../data/case_regression/words.txt','../data/case_regression/positive_data.jsonl','../data/case_regression/negative_data.jsonl')
	# import json
	# label2pmid = json.loads(open('./data/label2pmid/label2pmid_2310000.json','r').read())
	# num = []
	# for i, j in label2pmid.items():
	# 	num.append(len(j))
	pass
