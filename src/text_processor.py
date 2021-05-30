# Date: 2021-5-7
# Author: Qianqian Peng




def sent_tokenize(sent_tokenizer, text: str):
	sentence = sent_tokenizer(text)
	return sentence


def word_tokenize(word_tokenizer, sentence):
	if type(sentence) is str:
		return word_tokenizer(sentence)
	if type(sentence) is list:
		result = []
		for sent in sentence:
			result.append(word_tokenizer(sent))
		return result

def POS_tag(tagger, word):
	if type(word) is str:
		word = [word]
	return tagger(word)
