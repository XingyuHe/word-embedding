# %%
from timeit import main
from gensim.models import word2vec
from numpy.core.fromnumeric import shape
from scipy.sparse.coo import coo_matrix
from process import *
import numpy as np
import pandas as pd
import scipy.sparse
from scipy.sparse.linalg import svds
import pickle
import os
import logging
import subprocess

from gensim.test.utils import datapath
from gensim import utils
import gensim.models
import tempfile

# %%
FILE_NAME = "/Users/bobby/Documents/NLP/HW3/hw3/1_static_embeddings/data/brown.txt"
vocab = set()
idx_2_word = []
total_sentences = 0
for line in open(FILE_NAME, 'r').readlines():
    splits = line.strip().split()
    total_sentences += 1

    for word in splits:
        # TODO: think about more preprocessing stuff to do
        # maybe get rid of the punctuation? 
        word = word.lower()
        if word not in vocab:
            vocab.add(word) 
            idx_2_word.append(word)

word_2_idx = {word: idx for idx, word in enumerate(idx_2_word)}

# global variables
V = len(vocab)
N = total_sentences
MATRIX_DIR = "matrix"
# Parameters for the model
CONTEXT_WINDOW_SIZE = 10
VECTOR_SIZE = 100

# matrix directory contains all matrix calculated
if not os.path.isdir(MATRIX_DIR):
    print('The matrix directory is not present. Creating a new one..')
    os.mkdir(MATRIX_DIR)

CONTEXT_DIR = os.path.join(MATRIX_DIR, "context_{}".format(CONTEXT_WINDOW_SIZE))
# context directory context length
if not os.path.isdir(CONTEXT_DIR):
    print('The context directory is not present. Creating a new one..')
    os.mkdir(CONTEXT_DIR)

SAVE_DIR = CONTEXT_DIR


# %%
def row_broadcast_add(R, rv):
    """Sparse matrix plus a row vector"""
    R = R.tocsr()
    rv = rv.reshape(-1, 1)
    R.data += np.array(np.take(rv, R.indices)).reshape(-1)
    return R.tocoo()

def col_broadcast_add(R, cv):
    """Sparse matrix plus a column vector"""
    R = R.tocsc()
    cv = cv.reshape(1, -1)
    R.data += np.array(np.take(cv, R.indices)).reshape(-1)
    return R.tocoo()

# %%
def get_co_matrix_pmi_matrix():
    """[summary]
    Returns:
        [co-occurency matrix, pmi_matrix]: Return the co-occurency matrix and
                                         pmi matrix as a tuple
    """    
    co_matrix_file_name = os.path.join(SAVE_DIR, "co_matrix.pkl")
    pmi_matrix_file_name = os.path.join(SAVE_DIR, "pmi_matrix.pkl")
    
    if os.path.isfile(co_matrix_file_name):
        print("co_matrix is saved, reloading it ... ")
        with open(co_matrix_file_name, "rb") as file:
            co_matrix = pickle.load(file)

    else: 
        print("co_matrix is not saved, constructing it ...")
        co_matrix = scipy.sparse.lil_matrix((len(vocab), len(vocab)), dtype=int)

        for line in open(FILE_NAME, 'r').readlines():
            splits = list(map(lambda wrd: wrd.lower(), line.strip().split()))

            for idx, word in enumerate(splits):

                for context_idx in range(max(0, idx - CONTEXT_WINDOW_SIZE), min(len(splits), idx + CONTEXT_WINDOW_SIZE + 1)):
                    if context_idx == idx:
                        continue

                    context_word = splits[context_idx]
                    co_matrix[word_2_idx[word], word_2_idx[context_word]] += 1

        with open(co_matrix_file_name, "wb") as file:
            pickle.dump(co_matrix.tocoo(True), file)

    print("Finished construction the co-occurence matrix")
    co_matrix = co_matrix.tocoo()

    # get pmi_matrix
    if os.path.isfile(pmi_matrix_file_name):

        print("pmi_matrix is saved, reloading it ... ")
        with open(pmi_matrix_file_name+ "_normal_matrix", "rb") as file:
            pmi_matrix = pickle.load(file)

    else:
        print("ConsVtructing pmi matrix")
        marg_count = co_matrix.sum(axis=0)
        print('margin_count shape', marg_count.shape)
        total_count = marg_count.sum()
        print(total_count)

        join_prob = co_matrix / total_count # QUESTION: count with combination, not permutation? 
        join_prob.data = np.log(join_prob.data)
        print("join_prob shape", join_prob.shape)

        marg_prob = marg_count / total_count
        marg_prob.data = np.log(marg_prob.data)
        print("marg_prob_mult shape", marg_prob.shape)

        # Question:
        # Are there ways to broadcast subtraction for sparse matrix
        # Proper way to do log
        # What's the difference between them? 
        log_diff = col_broadcast_add(row_broadcast_add(join_prob, - marg_prob), 
                                        - marg_prob)
        log_diff.data = np.maximum(0, log_diff.data)
        pmi_matrix = log_diff

        print("logdiff shape", log_diff.shape)
        with open(pmi_matrix_file_name + "_normal_matrix", "wb") as file:
            pickle.dump(log_diff, file)

        print("Finished construction the pmi matrix")
        with open(pmi_matrix_file_name, "wb") as file:
            pickle.dump(pmi_matrix, file)

    return co_matrix, pmi_matrix
    

# %%
def get_svds():
    """[summary]

    Returns:
        [Dict]: return a svds dict consists of u, s, vT
                svds_dict = {"u": u,
                            "s": s,
                            "vT": vT}
    """
    svds_file_name = os.path.join(SAVE_DIR, 
                                "vector_size_{}_svds_dict_cont.plk".format(VECTOR_SIZE))
    if os.path.isfile(svds_file_name):
        print("svds is saved, reloading it ... ")

        with open(svds_file_name, "rb") as file:
            return pickle.load(file)

    print("svds is not saved, constructing it ... ")

    _, pmi_matrix = get_co_matrix_pmi_matrix()
    u, s, vT = scipy.sparse.linalg.svds(pmi_matrix, k=VECTOR_SIZE)
    svds_dict = {"u": u,
                "s": s,
                "vT": vT}
    with open(svds_file_name, "wb") as file:
        pickle.dump(svds_dict, file)

    return svds_dict


def get_embeddings():
    """
    Returns:
        [(word embedding, context embedding)]
    """    
    svds_dict = get_svds()
    u, s, vT = svds_dict["u"], svds_dict["s"], svds_dict["vT"]

    tempS = np.zeros(shape=(u.shape[-1], vT.shape[0]))
    np.fill_diagonal(tempS, s)
    s = tempS

    w = np.matmul(u, np.sqrt(s))
    c = np.matmul(vT.T, np.sqrt(s))

    embedding_file_name = os.path.join(SAVE_DIR, "embedding.txt")
    w_str = np.ndarray.astype(w, dtype=str)
    with open(embedding_file_name, "w") as file:
        file.writelines(str(idx_2_word[rowIdx]) + " " + " ".join(w_str[rowIdx, :]) + "\n" for rowIdx in range(w.shape[0]))

    print("The path to embedding from svd is \n {}".format(os.path.abspath(embedding_file_name)))
    model_path = os.path.abspath(embedding_file_name)
    return model_path
# %%
# word2vec training
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class MyCorpus:
    """An iterator that yields sentences (lists of str)."""

    def __iter__(self):
        corpus_path = datapath("/Users/bobby/Documents/NLP/HW3/hw3/1_static_embeddings/data/brown.txt")
        for line in open(corpus_path):
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(line)

def get_word2vec():
    MODEL_DIR = "./model"
    W2V_FILE_PATH = os.path.join(MODEL_DIR, "vector_size_{}_word2vec.kv".format(
                                                                    VECTOR_SIZE))
    if not os.path.isdir(MODEL_DIR):
        os.mkdir("model")

    if os.path.isfile(W2V_FILE_PATH):
        # To load a saved model:
        model_path = os.path.abspath(W2V_FILE_PATH)
        return model_path


    sentences = MyCorpus()
    # parameters for training
    k = 100
    min_count = 10
    workers = 10

    model = gensim.models.Word2Vec(sentences=sentences,
                                min_count=min_count,
                                vector_size=VECTOR_SIZE,
                                workers=workers)
    model.train(sentences, epochs=100, total_examples=total_sentences)

    # To save the model:
    model.save(W2V_FILE_PATH)
    print("The path to embedding from word2vec is \n {}".format(
                                os.path.abspath(W2V_FILE_PATH)))

    model_path = os.path.abspath(W2V_FILE_PATH)
    return model_path


if __name__ == "__main__":
    print()
    svd_embedding_path = get_embeddings()
    word2vec_path = get_word2vec()
    print("=========================== svds embedding ===================================")
    subprocess.run(["python", "evaluate.py", svd_embedding_path])
    print("============================ word2vec ================================")
    print(word2vec_path)
    subprocess.run(["python", "evaluate.py", word2vec_path])
# %%
