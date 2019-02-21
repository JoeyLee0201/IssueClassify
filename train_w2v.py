# -*- coding: UTF-8 -*-

from gensim.models import word2vec
import logging


def train(sentences, output_name):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # sentences = word2vec.Text8Corpus(u"./output/issue_corpus_tokenize.ic")

    model = word2vec.Word2Vec(sentences, size=100, sg=0, iter=150)

    model.save(output_name)

