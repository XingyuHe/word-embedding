# %%
'''
Please run this file in the 2_bert_
'''
import enum
import os
import argparse
from numpy.typing import _256Bit
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pickle
import sys
import time
from torch.nn.modules.pooling import AvgPool1d
import itertools

from transformers import BertTokenizer, BertModel
from sklearn.linear_model import LogisticRegression
from sklearn import svm

import multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool, Value
import transformers
from transformers.tokenization_utils_base import BatchEncoding
from tokenizers import Encoding as EncodingFast

from transformers.utils.dummy_pt_objects import Trainer
from transformers.utils.dummy_tokenizers_objects import BertTokenizerFast

import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import classification_report
'''
5428 number of samples
N: length of the sequence given to the BERT model
D: 768, dimension of the embedding
Out of the BERT for each sentence should be 

BERT: bert-base-cased
'''

# %%
def get_wic_subset(data_dir):
	wic = []
	LABELS = ['F', 'T']
	split = data_dir.strip().split('/')[-1]
	with open(os.path.join(data_dir, '%s.data.txt' % split), 'r', encoding='utf-8') as datafile, \
		open(os.path.join(data_dir, '%s.gold.txt' % split), 'r', encoding='utf-8') as labelfile:
		# open the data and gold data
		for (data_line, label_line) in zip(datafile.readlines(), labelfile.readlines()):
			# col: word to be tested, POS, the position at each sentence
			word, _, word_indices, sentence1, sentence2 = data_line.strip().split('\t')
			sentence1_word_index, sentence2_word_index = word_indices.split('-')

			# convert F and T to number
			label = LABELS.index(label_line.strip())
			wic.append({
				'word': word,
				'sentence1_word_index': int(sentence1_word_index),
				'sentence2_word_index': int(sentence2_word_index),

				# changed original way to tokenize
				# 'sentence1_words': sentence1.split(' '),
				# 'sentence2_words': sentence2.split(' '),

				# changed original way to tokenize
				'sentence1_words': sentence1,
				'sentence2_words': sentence2,
				'label': label
			})
	# wic is a list of dictionary where each dictionary is an instance of training
	return wic

# %%
if __name__ == '__main__':
	# change the project root if you are running on a different machine
	fn_start_time = time.time()
	SECTION2_ROOT = os.getcwd()
	print("Current directory root is", SECTION2_ROOT)
	SECTION2_ROOT = os.path.join(SECTION2_ROOT, "hw3/2_bert_wic/")

	# Load data
	TRAIN_DATA_DIR = os.path.join(SECTION2_ROOT, "wic", "train")
	wic_train = get_wic_subset(TRAIN_DATA_DIR)
	df_wic_train = pd.DataFrame(wic_train)

	TEST_DATA_DIR = os.path.join(SECTION2_ROOT, "wic", "dev")
	wic_test = get_wic_subset(TEST_DATA_DIR)
	df_wic_test = pd.DataFrame(wic_test)

	# Load BERT
	tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
	model = BertModel.from_pretrained('bert-base-cased')
	model = model.eval()

	with torch.no_grad():
		train_sent = list(df_wic_train['sentence1_words']) + list(df_wic_train['sentence2_words'])
		token = tokenizer(train_sent, padding=True, return_tensors='pt')
		embed = model(**token)

		with open('embed_train.plk', 'wb') as file:
			pickle.dump(embed, file)

		test_sent = list(df_wic_test['sentence1_words']) + list(df_wic_test['sentence2_words'])
		token = tokenizer(test_sent, padding=True, return_tensors='pt')
		embed = model(**token)

		with open('embed_test.plk', 'wb') as file:
			pickle.dump(embed, file)

	print("run time : {}".format(time.time() - fn_start_time))