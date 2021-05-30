# Date: 2021-5-29
# Author: Qianqian Peng

import os
inpath = '../data/label_model/'
# inpath = '../data/label_model_fixFeature/'
dirlist = os.listdir(inpath)
acc_lg = []
acc_svm = []
n = 0
m=0
for dir in dirlist:
	one_model_acc_path = inpath+'/'+dir+'/acc_sklearn.txt'

	acc = open(one_model_acc_path,'r').read().split('\n')[0:2]
	acc_lg.append(float(acc[0].split(': ')[1]))
	acc_svm.append(float(acc[1].split(': ')[1]))
	if acc_lg[-1]>0.98 or acc_svm[-1]>0.98:
		print(dir)
		m+=1
	n += 1
	if n % 100 == 0:
		print("process {}/{}".format(n,len(dirlist)))
print(sum(acc_lg)/len(acc_lg))
print(sum(acc_svm)/len(acc_svm))

#'Tobacco mosaic satellite virus'
