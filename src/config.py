# Date: 2021-4-13
# Author: Qianqian Peng


class ConfigClassifier:
	def __init__(self,clf=True):
		self.bert_path = '../model/biobert-base-cased'
		self.bert_out_dim = 768
		self.n_epoch = 30
		self.batch_size = 64
		self.device = 'cuda'
		self.lr = 2e-5
		self.onlyclf = clf
		self.class_num = 29641
		if self.onlyclf:
			self.model_save_path = '../model/Classifier/'
			self.evaluate_model = '../model/Classifier/model_30.bin'
			self.out_path = '../data/test_data/classifier_result_selected/'

		else:
			# bi-task model
			self.model_save_path = '../model/BiTaskModel/'
			self.evaluate_model = '../model/BiTaskModel/model_30.bin'
			self.out_path = '../data/test_data/bitask_result_selected/'

		self.evaluate_data = '../data/test_data/Articles_PubTator_selected/'
		self.train_data = '../data/train_data/distant_train_data/distant_train.conll'
		self.dict_path = '../dict/'

		self.device_n = 0
		self.save_step = 4
		self.pre_compute_words_embed = True
		self.words_embed = '../model/embedding/word_embed_for_biTaskModel.t7'


class ConfigBiEncoder:
	def __init__(self):
		self.bert_path = '../model/biobert-base-cased'
		self.bert_out_dim = 768
		self.n_epoch = 30
		self.batch_size = 32
		self.device = 'cuda'
		self.lr = 1e-5
		self.model_save_path = '../model/BiEncoder/'
		self.evaluate_model = '../model/BiEncoder/model_30.bin'
		self.evaluate_data = '../data/test_data/Articles_PubTator_selected/'
		self.out_path = '../data/test_data/biencoder_result_selected/'
		self.train_data = '../data/train_data/distant_train_data/distant_train.conll'
		self.dict_path = '../dict/'
		self.device_n = 0
		self.parall = False
		self.save_step = 4
		self.max_batches = 1024
		self.from_checkpoint = True
		if self.from_checkpoint:
			self.checkpoint_step = 30
		self.pre_compute_words_embed = True
		self.words_embed = '../model/embedding/word_embed.t7'



# class a():
# 	def __init__(self):
# 		self.save_step=4
# b=a()
# print(b.save_step)
# print(type(b.save_step))
