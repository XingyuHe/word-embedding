# %%
import enum
import os
import argparse
from sklearn.utils.extmath import fast_logdet
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pickle
import sys
import time
from torch.nn.modules.pooling import AvgPool1d, AvgPool2d
import itertools
import collections

from sklearn.metrics import classification_report

from transformers import BertTokenizer, BertModel
from multiprocessing.dummy import Pool as ThreadPool, Value
import transformers
from transformers.tokenization_utils_base import BatchEncoding
from tokenizers import Encoding as EncodingFast

from transformers.utils.dummy_pt_objects import Trainer, torch_distributed_zero_first
from transformers.utils.dummy_tokenizers_objects import BertTokenizerFast

'''
5428 number of samples
N: length of the sequence given to the BERT model
D: 768, dimension of the embedding
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
# Get the embedding of the data
class GetFeatures:
	def __init__(self) -> None:
		pass

	def get_embedding_unpad(self, df_wic, tokenizer, model, train=True, save=False):

		if train:
			filename_embed1 = "train_embedding_sent1.plk"
			filename_embed2 = "train_embedding_sent2.plk"
		else:
			filename_embed1 = "test_embedding_sent1.plk"
			filename_embed2 = "test_embedding_sent2.plk"

		fn_start_time = time.time()

		# check if the embedding has been created
		if os.path.isfile(filename_embed1) and os.path.isfile(filename_embed2):
			print("the embeddings are found, loading them ... ")
			with open(filename_embed1, 'rb') as file1, open(filename_embed2, 'rb') as file2:
				return pickle.load(file1), pickle.load(file2)

		# thread worker in charge of calculating the embeddings for each sentence
		def get_embedding_worker(idx, sent):
			print(idx, len(sent))
			sys.stdout.flush()

			with torch.no_grad():
				input_sent = tokenizer(sent, return_tensors="pt")
				outputs = model(**input_sent)
				hidden = outputs.last_hidden_state
				# (1, #tokens, 768)
				return hidden.squeeze()
				# (#tokens, 768)

		print("the embeddings are not found, constructing them ... ")

		# calculate the embedding across all sentences: 2 ways
		# multi-threaded
		thread_count = 10
		with ThreadPool(thread_count) as pool:
			sent1 = list(df_wic["sentence1_words"])
			embed_sent1 = pool.starmap(get_embedding_worker, enumerate(sent1))
			sent2 = list(df_wic["sentence2_words"])
			embed_sent2 = pool.starmap(get_embedding_worker, enumerate(sent2))

		# single-threaded
		# sent1 = list(df_wic["sentence1_words"])
		# embed_sent1 = list(itertools.starmap(get_embedding_worker, enumerate(sent1)))
		# sent2 = list(df_wic["sentence2_words"])
		# embed_sent2 = list(itertools.starmap(get_embedding_worker, enumerate(sent2)))
		

		if save:
			with open(filename_embed1, 'wb') as file1:
				pickle.dump(embed_sent1, file1)
			with open(filename_embed2, 'wb') as file2:
				pickle.dump(embed_sent2, file2)

		print("get_embedding_unpad took {} to run".format(time.time() - fn_start_time))

		return embed_sent1, embed_sent2
		# List[(#token, 768)], List[(#token, 768)]

	def get_feature_max_pool_embedding(self, df_wic, tokenizer, model, train):
		'''
			Pool word embedding across words in a sentence with average
		'''
		embed_sent1, embed_sent2 = self.get_embedding_unpad(df_wic, tokenizer, model, train)

		avg_pool_sent1 = []
		for embed in embed_sent1:
			avg_pool_sent1.append(torch.amax(embed, 0))
		avg_pool_sent1 = torch.stack(avg_pool_sent1)

		avg_pool_sent2 = []
		for embed in embed_sent2:
			avg_pool_sent2.append(torch.amax(embed, 0))
		avg_pool_sent2 = torch.stack(avg_pool_sent2)

		return torch.concat([avg_pool_sent1, avg_pool_sent2], 1)


	def get_feature_average_pool_embedding(self, df_wic, tokenizer, model, train):
		'''
			Pool word embedding across words in a sentence with average
		'''
		embed_sent1, embed_sent2 = self.get_embedding_unpad(df_wic, tokenizer, model, train)

		avg_pool_sent1 = []
		for embed in embed_sent1:
			avg_pool_sent1.append(torch.mean(embed, 0))
			#avg_pool dim [768]
			# embed has dim [#tokens, 768]

		avg_pool_sent1 = torch.stack(avg_pool_sent1)
		# [#sent, 768]

		avg_pool_sent2 = []
		for embed in embed_sent2:
			avg_pool_sent2.append(torch.mean(embed, 0))
		avg_pool_sent2 = torch.stack(avg_pool_sent2)

		# [#sent, 2 * 768]
		# return avg_pool_sent1 - avg_pool_sent2
		return torch.concat([avg_pool_sent1, avg_pool_sent2], 1)


	def get_feature_target_word_embedding(self, df_wic, tokenizer, model, train):
		"""[summary]
			create features by calculating the dot product between 
			embedding of the target word from two different sentences
		"""	

		embed_sent1, embed_sent2 = self.get_embedding_unpad(df_wic, tokenizer, model, train)

		df_wic_word_sent1_idx = df_wic['sentence1_word_index']
		ts_sent1 = list(df_wic["sentence1_words"])
		token_encoding_sent1: BatchEncoding = tokenizer(ts_sent1, return_tensors="np")

		df_wic_word_sent2_idx = df_wic['sentence2_word_index']
		ts_sent2 = list(df_wic["sentence2_words"])
		token_encoding_sent2: BatchEncoding = tokenizer(ts_sent2, return_tensors="np")

		def get_embed_idx_worker(batch_idx):

			embed1_idx = token_encoding_sent1.word_to_tokens(batch_idx, df_wic_word_sent1_idx[batch_idx])
			embed2_idx = token_encoding_sent2.word_to_tokens(batch_idx, df_wic_word_sent2_idx[batch_idx])

			return (embed1_idx.start, embed1_idx.end), (embed2_idx.start, embed2_idx.end)

		target_embed_1 = []
		target_embed_2 = []

		for batch_idx in range(len(embed_sent1)):

			token1_idx, token2_idx = get_embed_idx_worker(batch_idx)

			token1_embedding = embed_sent1[batch_idx][token1_idx[0]: token1_idx[1], :]
			token1_pool_embed = torch.mean(token1_embedding, 0)
			target_embed_1.append(token1_pool_embed)

			token2_embedding = embed_sent2[batch_idx][token2_idx[0]: token2_idx[1], :]
			token2_pool_embed = torch.mean(token2_embedding, 0)
			target_embed_2.append(token2_pool_embed)

		cosine_similarity = []
		target_embed_cat = []
		target_embed_diff = []
		for embed1, embed2 in zip(target_embed_1, target_embed_2):
			cosine_similarity.append(torch.dot(embed1, embed2) / 
									(torch.linalg.norm(embed1) * torch.linalg.norm(embed2)))
			target_embed_cat.append(torch.concat([embed1, embed2], 0))
			target_embed_diff.append(embed1 - embed2)

		# return torch.stack(target_embed_cat)
		return torch.stack(target_embed_diff)
		# return torch.stack(cosine_similarity).unsqueeze(1)

	def get_feature_cosine_similarity_average_pool(self, df_wic, tokenizer, model, train):
		pool_embed = self.get_feature_average_pool_embedding(df_wic, tokenizer, model, train)
		num_feat = pool_embed.shape[1] // 2
		pool_embed1, pool_embed2 = pool_embed[:, :num_feat], pool_embed[:, num_feat:]
		norm1, norm2 = torch.linalg.norm(pool_embed1, dim=1), torch.linalg.norm(pool_embed2, dim=1)
		dot_prod = torch.diagonal(torch.matmul(pool_embed1, pool_embed2.T)).unsqueeze(1)
		return dot_prod / (norm1 @ norm2)


class EmbeddingOutputLayer(torch.nn.Module):
	def __init__(self, embed_feat_dim, dropout_rate) -> None:
		super(EmbeddingOutputLayer, self).__init__()

		self.model = torch.nn.Sequential(
			collections.OrderedDict([
				('dense0', torch.nn.Linear(embed_feat_dim, 300)),
				('batchNorm0', torch.nn.BatchNorm1d(300)),
				('relu0', torch.nn.ReLU()),
				('dense1', torch.nn.Linear(300, 100)),
				('batchNorm1', torch.nn.BatchNorm1d(100)),
				('relu1', torch.nn.ReLU()),
				('dropout1', torch.nn.Dropout(dropout_rate)),
				('dense2', torch.nn.Linear(100, 50)),
				('batchNorm2', torch.nn.BatchNorm1d(50)),
				('relu2', torch.nn.ReLU()),
				# ('dropout2', torch.nn.Dropout(dropout_rate)),
				('dense3', torch.nn.Linear(50, 2)),
				('output', torch.nn.Softmax(1))
			])
		)

	def forward(self, x):
		return self.model(x)

def train(train_data, test_data, model, optimizer, loss_fn, patience):

	test_loss_history = [1000] * patience
	test_loss = 0

	def closure():
		optimizer.zero_grad()
		y_pred = model(X)
		loss = loss_fn(y_pred, y)
		loss.backward()
		return loss
	
	
	epoch = 0
	while test_loss < max(test_loss_history[-patience + 1:]):
		X, y = train_data
		optimizer.step(closure)

		with torch.no_grad():
			X, y = test_data
			test_loss = loss_fn(model(X), y)
			X, y = train_data
			train_loss = loss_fn(model(X), y)

		if epoch % 100 == 0:
			print("Epcoh: {}, Train loss: {}, Test loss: {}".format(epoch, train_loss, test_loss))
		test_loss_history.append(test_loss)
		epoch += 1

	return min(test_loss_history)


def main(args, config=None):

	# Getting data
	torch.manual_seed(0)

	# Load data
	TRAIN_DATA_DIR = args.train_dir
	wic_train = get_wic_subset(TRAIN_DATA_DIR)
	df_wic_train = pd.DataFrame(wic_train)

	TEST_DATA_DIR = args.eval_dir
	wic_test = get_wic_subset(TEST_DATA_DIR)
	df_wic_test = pd.DataFrame(wic_test)

	# Load BERT
	# tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
	fastTokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-cased')
	model = BertModel.from_pretrained('bert-base-cased')
	model = model.eval()

	# Get features
	getFeatures = GetFeatures()

	train_target_embed = getFeatures.get_feature_target_word_embedding(df_wic_train, fastTokenizer, model, train=True)
	test_target_embed = getFeatures.get_feature_target_word_embedding(df_wic_test, fastTokenizer, model, train=False)

	train_data = (train_target_embed, torch.tensor(df_wic_train['label']))
	test_data = (test_target_embed, torch.tensor(df_wic_test['label']))

	# building model and hyper parameter
	if not config:
		config = {
			"learning_rate": 0.01,
			"patience": 200,
			"weight_decay": 1e-5,
			"dropout": 0.5
		}

	clf = EmbeddingOutputLayer(train_data[0].shape[1], config['dropout'])
	optimizer = torch.optim.Adam(clf.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
	loss_fn = torch.nn.CrossEntropyLoss()

	# train and test, produce classification reports
	lowest_loss = train(train_data, test_data, clf, optimizer, loss_fn, config['patience'])
	
	train_y_softmax = clf(train_data[0])
	train_y_pred = torch.argmax(train_y_softmax, 1)
	test_y_softmax = clf(test_data[0])
	test_y_pred = torch.argmax(test_y_softmax, 1).numpy()

	print(
		classification_report(df_wic_train['label'],
								train_y_pred)
	)
	print(
		classification_report(df_wic_test['label'],
								test_y_pred)
	)
	result = classification_report(df_wic_test['label'], test_y_pred, output_dict=True)
	print('accuracy', result['accuracy'])

	with open(args.out_file, "w") as file:
		num_to_TF = {0: "F\n", 1: "T\n"}
		file.writelines(map(lambda pred: num_to_TF[pred], test_y_pred))

	return lowest_loss, result

def grid_search(args):
	# 0.64
	# config = {
	# 	"learning_rate": 0.01,
	# 	"patience": 200,
	# 	"weight_decay": 1e-5,
	# 	"dropout": 0.5
	# }

	# accuracy 0.628
	# config = {
	# 	"learning_rate": 0.01,
	# 	"patience": 50,
	# 	"weight_decay": 1e-5,
	# 	"dropout": 0.7
	# }

	# # accuracy 0.624
	# config = {
	# 	"learning_rate": 0.001,
	# 	"patience": 100,
	# 	"weight_decay": 1e-5,
	# 	"dropout": 0.3
	# }
	CONFIG = {
		"learning_rate": [10 ** (-n) for n in range(2, 5)],
		"patience": [10, 20, 50, 80, 100, 150, 200],
		"weight_decay": [0.01, 0.05, 0.005, 0.001],
		"dropout": [i / 10 for i in range(1, 10)]
	}

	lowest_test_loss = float('inf')
	lowest_config = None
	lowest_res = None
	lowest_model = None


	for learning_rate in CONFIG['learning_rate']:
		for patience in CONFIG['patience']:
			for weight_decay in CONFIG['weight_decay']:
				for dropout in CONFIG['dropout']:
					config = {
						"learning_rate": learning_rate,
						"patience": patience,
						"weight_decay": weight_decay,
						"dropout": dropout,
					}
					print("config", config)
					loss, result = main(args, config)
					print("loss", loss)

					if lowest_test_loss > loss:
						lowest_config, lowest_test_loss, lowest_res, lowest_model = config, loss, result, model

	return  lowest_test_loss, lowest_config, lowest_res, lowest_model



if __name__ == '__main__':
	start_time = time.time()

	parser = argparse.ArgumentParser(
		description='Train a classifier to recognize words in context (WiC).'
	)
	parser.add_argument(
		'--train-dir',
		dest='train_dir',
		required=True,
		help='The absolute path to the directory containing the WiC train files.'
	)
	parser.add_argument(
		'--eval-dir',
		dest='eval_dir',
		required=True,
		help='The absolute path to the directory containing the WiC eval files.'
	)
	# Write your predictions (F or T, separated by newlines) for each evaluation
	# example to out_file in the same order as you find them in eval_dir.  For example:
	# F
	# F
	# T
	# where each row is the prediction for the corresponding line in eval_dir.
	parser.add_argument(
		'--out-file',
		dest='out_file',
		required=True,
		help='The absolute path to the file where evaluation predictions will be written.',
	)
	args = parser.parse_args()

	print(
		"arguments are \n train_dir: {}\n test_dir: {}\n outfile: {}\n".format(
			args.train_dir, args.eval_dir, args.out_file
		)
	)

	main(args)
	print("Program took {} s".format(time.time() - start_time))
