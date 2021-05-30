# Date: 2021-5-24
# Author: Qianqian Peng

import json
import random
import sys,os,time
import torch
import torch.nn as nn
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn import feature_selection as FS
from joblib import dump,load
from sklearn import linear_model,svm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

def pop_tail(vec,tail_c=''):
	if vec[-1] == tail_c:
		vec.pop(-1)

def check_label(data_path='F:/Data/allMeSH_2020/allMeSH_2020.json',
                dict_path='../dict/all_name_id.txt',
                out='../data/case_regression/mesh_covered_sample.txt'):
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
            n += 1
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


def build_trainData(pmids,articles,data_path, label2pmid, target_label='Kidney Glomerulus',
                    max_train=100, percent=2/3,out_path='../data/data_ML/'):
    articles_num = len(pmids)
    target_articles_num = len(label2pmid[target_label])
    target_pmids = label2pmid[target_label]
    train_pos_pmid = target_pmids[0:int(target_articles_num*percent)][0:max_train]
    train_neg_pmid = []

    test_pos_pmid = target_pmids[int(target_articles_num*percent):][0:max_train]
    test_neg_pmid = []

    n = 0
    while n < len(train_pos_pmid):
        rand_idx = random.randint(0, articles_num - 1)
        while pmids[rand_idx] in target_pmids:
            rand_idx = random.randint(0, articles_num - 1)
        train_neg_pmid.append(pmids[rand_idx])
        n += 1
    n = 0
    while n < len(test_pos_pmid):
        rand_idx = random.randint(0, articles_num - 1)
        while pmids[rand_idx] in target_pmids or pmids[rand_idx] in train_neg_pmid:
            rand_idx = random.randint(0, articles_num - 1)
        test_neg_pmid.append(pmids[rand_idx])
        n += 1
    train_pos_articls = [articles[pmids.index(p)] for p in train_pos_pmid]
    train_neg_articls = [articles[pmids.index(p)] for p in train_neg_pmid]
    test_pos_articls = [articles[pmids.index(p)] for p in test_pos_pmid]
    test_neg_articls = [articles[pmids.index(p)] for p in test_neg_pmid]
    out_path = out_path + '/' + target_label + '/'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    pos_pmid_train = open(out_path + '/' + 'pos_pmid_train.txt', 'w', encoding='utf-8', newline='\n')
    neg_pmid_train = open(out_path + '/' + 'neg_pmid_train.txt', 'w', encoding='utf-8', newline='\n')
    pos_pmid_test = open(out_path + '/' + 'pos_pmid_test.txt', 'w', encoding='utf-8', newline='\n')
    neg_pmid_test = open(out_path + '/' + 'neg_pmid_test.txt', 'w', encoding='utf-8', newline='\n')

    positive_out_train = open(out_path + '/positive_data_train.txt', 'w', encoding='utf-8', newline='\n')
    negative_out_train = open(out_path + '/negative_data_train.txt', 'w', encoding='utf-8', newline='\n')
    positive_out_test = open(out_path + '/positive_data_test.txt', 'w', encoding='utf-8', newline='\n')
    negative_out_test = open(out_path + '/negative_data_test.txt', 'w', encoding='utf-8', newline='\n')

    pos_pmid_train.write('\n'.join(train_pos_pmid))
    pos_pmid_train.close()

    neg_pmid_train.write('\n'.join(train_neg_pmid))
    neg_pmid_train.close()

    pos_pmid_test.write('\n'.join(test_pos_pmid))
    pos_pmid_test.close()

    neg_pmid_test.write('\n'.join(test_neg_pmid))
    neg_pmid_test.close()

    positive_out_train.write('\n'.join(train_pos_articls))
    positive_out_train.close()

    negative_out_train.write('\n'.join(train_neg_articls))
    negative_out_train.close()

    positive_out_test.write('\n'.join(test_pos_articls))
    positive_out_test.close()

    negative_out_test.write('\n'.join(test_neg_articls))
    negative_out_test.close()





def combine_pos_neg(path):
    out = path + '/data_train.txt'
    out = open(out, 'w', encoding='utf-8', newline='\n')
    with open(path + 'positive_data_train.txt', 'r', encoding='iso8859') as f:
        line = f.read()
        out.write(line)

    with open(path + '/negative_data_train.txt', 'r', encoding='iso8859') as f:
        line = f.read()
        out.write(line)
    out.close()


def build_testData(pmids,articles,data_path,
                   label2pmid: dict,
                   pos_pmid_train_path,
                   neg_pmid_train_path,
                   target_label='Kidney Glomerulus',
                   max_num=400,
                   percent=1/3,
                   out_path='./'):
    pos_pmid_train = open(pos_pmid_train_path, 'r').read().split('\n')
    neg_pmid_train = open(neg_pmid_train_path, 'r').read().split('\n')
    target_pmid = label2pmid[target_label]
    if percent is not None:
        max_num2 = int(len(target_pmid)*percent)
        if max_num > max_num2:
            max_num = max_num2
    print(len(target_pmid))
    print(max_num)
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
            line = line.strip()
            n += 1
            record = json.loads(line)
            pmid = record['pmid']
            if target_pmid_dict.get(pmid) is not None and pos_train_dict.get(pmid) is None:
                positive_out.write(line + '\n')
                num_pos += 1
                if num_pos == max_num:
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


def feature_selection(words_path, positive_path, negtive_path,fixFeature=None):
    lemmatizer = WordNetLemmatizer()
    all_words = open(words_path, 'r', encoding='iso8859').read().split('\n')
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
        one_pos_record = pos_text[i]
        
        one_pos_words = nltk.word_tokenize(one_pos_record)
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
        one_neg_record = neg_text[i]
        one_neg_words = nltk.word_tokenize(one_neg_record)
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
    if fixFeature is not None:
        skb = SelectKBest(score_func=FS.f_classif, k=fixFeature)
    else:
        skb = SelectKBest(score_func=FS.f_classif, k=int(len(pos_feature[0]) * 0.01))
    new_x = skb.fit_transform(pos_feature + neg_feature, pos_label + neg_label)
    select_idx = skb.get_support()
    selected_words = []
    for i in range(len(select_idx)):
        if select_idx[i] == True:
            selected_words.append(all_words[i])
    return all_words,selected_words



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
            text = line.strip()
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
            text = line.strip()
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
                           feature_words_path,model_out_path):
    train_data, train_label = load_data(pos_path, neg_path, feature_words_path)
    test_data, test_label = load_data(pos_path_test, neg_path_test, feature_words_path)
    print('build training data done.')
    logistic_model = LogisticRegression(len(train_data[0]))
    svm_model = SVM(len(train_data[0]))
    # sklearn model
    logreg = linear_model.LogisticRegression(C=1e3, solver='liblinear')
    logreg.fit(train_data, train_label)
    svm_sk = svm.SVC()
    svm_sk.fit(train_data, train_label)
    device='cpu'
    #if torch.cuda.is_available():
        #logistic_model.to('cuda')
        #svm_model.to('cuda')

    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(logistic_model.parameters(), lr=1e-5, momentum=0.9)
    optimizer_svm = torch.optim.SGD(svm_model.parameters(), lr=1e-5, momentum=0.9)
    max_acc = -1000
    max_acc_svm = -1000
    for _ in range(20):
        rand_idx = list(range(len(train_data)))
        random.shuffle(rand_idx)
        train_data_rand = [train_data[i] for i in rand_idx]
        train_label_rand = [train_label[i] for i in rand_idx]
        step = 64
        n = 0
        acc_all = []
        acc_all_svm = []
        
        while n < len(train_data_rand):
            x_data = train_data_rand[n:n + step]
            y_data = train_label_rand[n:n + step]
            x_data = torch.tensor(x_data).float().to(device)
            y_data = torch.tensor(y_data).float().unsqueeze(-1).to(device)
            out = logistic_model(x_data)
            loss = criterion(out, y_data)
            out_svm = svm_model(x_data)
            loss_svm = svm_model.compute_loss(out_svm, y_data)
            print_loss = loss.data.item()
            # print('loss: ',print_loss)
            mask = out.ge(0.5).float()  # 以0.5为阈值进行分类
            mask_svm = out_svm.ge(0.5).float()  # 以0.5为阈值进行分类
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
        # print('*' * 20)
        # print('train overall acc: ', sum(acc_all) / len(acc_all))
        # print('train overall acc_svm: ', sum(acc_all_svm) / len(acc_all_svm))
        # for test model
        with torch.no_grad():
            x_data = torch.tensor(test_data).float().to(device)
            y_data = torch.tensor(test_label).float().unsqueeze(-1).to(device)
            out = logistic_model(x_data)
            out_svm = svm_model(x_data)
            mask = out.ge(0.5).float()  # 以0.5为阈值进行分类
            mask_svm = out_svm.ge(0.5).float()  # 以0.5为阈值进行分类
            correct = (mask == y_data).sum()  # 计算正确预测的样本个数
            correct_svm = (mask_svm == y_data).sum()  # 计算正确预测的样本个数
            acc = correct.item() / x_data.size(0)
            acc_svm = correct_svm.item() / x_data.size(0)

            if acc > max_acc:
                max_acc = acc
                torch.save(logistic_model.state_dict(),model_out_path+'/logistic.bin')
            if acc_svm > max_acc_svm:
                max_acc_svm = acc_svm
                torch.save(svm_model.state_dict(), model_out_path + '/svm.bin')
    acc_sk = logreg.score(test_data, test_label)
    acc_svm_sk = svm_sk.score(test_data, test_label)
    print("test acc_logistic: ", max_acc)

    print("test acc_svm: ", max_acc_svm)
    print("test acc_log_sk: ", acc_sk)
    print("test acc_svm_sk: ", acc_svm_sk)
    with open(model_out_path+'/acc.txt','w',newline='\n') as f:
        f.write('logistic: '+str(max_acc)+'\n')
        f.write('svm: '+str(max_acc_svm)+'\n')
    with open(model_out_path+'/acc_sklearn.txt','w',newline='\n') as f:
        f.write('logistic: '+str(acc_sk)+'\n')
        f.write('svm: '+str(acc_svm_sk)+'\n')

    dump(logreg, model_out_path+'/logreg.joblib')
    dump(svm_sk, model_out_path+'/svm.joblib')


#check_label('/media/dragon/F/PYProject/diploma_project/data/sample_allMeSH_2020.jsonl')

#split_trainData_label('/media/dragon/F/PYProject/diploma_project/data/sample_allMeSH_2020.jsonl')
build_data = False
if build_data:
    print("load pmid2articels.")
    pmids = []
    texts = []
    with open('/media/dragon/F/PYProject/diploma_project/data/sample_allMeSH_2020.jsonl','r',encoding='iso8859') as f:
        n = 0
        for line in f:
            line =json.loads(line)
            title = line['title']
            if title is None:
                title = ''
            pmids.append(line['pmid'])
            texts.append(title+line['abstractText'])
            n += 1
            if n % 50000 == 0:
                print("load {} articels.".format(n))
    print("load articles done.")
    print("load label2pmid dict.")
    label2pmid = json.loads(open('../data/label2pmid_sample/label2pmid_3853918.json').read())
    print("load label2pmid dict done.")
    print("load covered label.")
    covered_label = open('../data/case_regression/mesh_covered_sample.txt').read().split('\n')
    if covered_label[-1] == '':
        covered_label.pop(-1)
    print("load covered label done.")
    data_path = '/media/dragon/F/PYProject/diploma_project/data/sample_allMeSH_2020.jsonl'
    out_dir = '../data/data_ML/'
    import os

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    n = 0
    import time
    few_train_label = []
    no_feature_words = []
    except_situation_label = []
    import warnings
    warnings.filterwarnings('ignore')

    # for build train and test data
    for label in covered_label:
        print('*'*20)
        print("label: ", label)
        n += 1
        if len(label2pmid[label]) < 6:
            few_train_label.append(label)
            continue

        if os.path.exists(out_dir + '/' + label):
            print(out_dir + '/' + label,' exits.')
            if not os.path.exists(out_dir+'/'+label+'/negative_data_test.txt'):
                build_trainData(pmids, texts, data_path, label2pmid, label, 100, 2 / 3, out_dir)

            continue


        print("process label: {}/{}".format(n, len(covered_label)))
        st = time.time()
        build_trainData(pmids,texts,data_path, label2pmid, label, 100, 2/3,out_dir)
        ed = time.time()
        print("build train: ",ed-st)
        combine_pos_neg(out_dir + '/' + label + '/')
        st = time.time()
        split_text(out_dir + '/' + label + '/data_train.txt', out_dir + '/' + label + '/words.txt')
        ed = time.time()
        print("split: ", ed - st)
        st = time.time()

        print('start train model done.')
        print("few train data label num: ",len(few_train_label))
        print("no feature word label num: ",len(no_feature_words))


    few_train_label_out = open('/media/dragon/F/PYProject/diploma_project/data/few_trainData_label.txt','w',encoding='utf-8',newline='\n')
    few_train_label_out.write('\n'.join(few_train_label))
    few_train_label_out.close()

    no_feature_label_out = open('/media/dragon/F/PYProject/diploma_project/data/no_feature_label.txt','w',encoding='utf-8',newline='\n')
    no_feature_label_out.write('\n'.join(no_feature_words))
    no_feature_label_out.close()

    except_out = open('/media/dragon/F/PYProject/diploma_project/data/except_label.txt','w',encoding='utf-8',newline='\n')
    except_out.write('\n'.join(except_situation_label))
    except_out.close()


train_model = True
if train_model:
    import warnings
    warnings.filterwarnings('ignore')
    model_out_path = '../data/label_model_fixFeature/'
    if not os.path.exists(model_out_path):
        os.mkdir(model_out_path)
    # for train model
    covered_label = open('../data/case_regression/mesh_covered_sample.txt').read().split('\n')
    if covered_label[-1] == '':
        covered_label.pop(-1)
    print("load covered label done.")
    trained_label = []
    n = 0
    has_trained_label = open('../data/trained_label_fixFeature.txt','r').read().split('\n')
    if has_trained_label[-1] == '':
        has_trained_label.pop(-1)
    for label in covered_label:
        if label in has_trained_label:
            n += 1
            print("process label: {}/{}".format(n, len(covered_label)))
            trained_label.append(label)
            continue
        print('*'*20)
        indir = '../data/data_ML/' + label + '/'
        if not os.path.exists(indir) or not os.path.exists(indir+'/negative_data_test.txt'):
            continue
        print("label: ", label)


        combine_pos_neg(indir)

        print('split the text to words.')
        split_text(indir+'/data_train.txt', indir+'/words.txt')
        print('split the text to words done.')

        print('start select feature.')
        words, selected_words = feature_selection(indir+'/words.txt',
                                                  indir+'/positive_data_train.txt',
                                                  indir+'/negative_data_train.txt',30)
        model_out_path_one_label = model_out_path + '/' + label
        if not os.path.exists(model_out_path_one_label):
            os.mkdir(model_out_path_one_label)
        out_feature_words = open(model_out_path_one_label+'/feature_words.txt', 'w', encoding='iso8859', newline='\n')
        out_feature_words.write('\n'.join(selected_words))
        out_feature_words.close()
        print('start select feature done.')
        st = time.time()

        logiatic_regress_train(indir+'/positive_data_train.txt',
                               indir+'/negative_data_train.txt',
                               indir+'/positive_data_test.txt',
                               indir+'/negative_data_test.txt',
                               indir+'/feature_words.txt',
                               model_out_path_one_label)
        ed = time.time()
        print("train model duration: ",ed-st)
        n += 1
        print("process label: {}/{}".format(n, len(covered_label)))
        trained_label.append(label)
        with open('../data/trained_label_fixFeature.txt','w',encoding='utf-8',newline='\n') as f:
            f.write('\n'.join(trained_label))




# if __name__ == '__main__':
#     # split_trainData_label('F:/Data/allMeSH_2020/allMeSH_2020.json')
#     # build_trainData('F:/Data/allMeSH_2020/allMeSH_2020.json')
#     # split_text('../data/case_regression/data.txt','../data/case_regression/words.txt')
#     # pv, fv = feature_selection('../data/case_regression/words.txt','../data/case_regression/positive_data.jsonl','../data/case_regression/negative_data.jsonl')
#     # import json
#     # label2pmid = json.loads(open('./data/label2pmid/label2pmid_2310000.json','r').read())
#     # num = []
#     # for i, j in label2pmid.items():
#     # 	num.append(len(j))
#     pass
