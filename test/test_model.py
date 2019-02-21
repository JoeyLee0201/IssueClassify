# -*- coding: UTF-8 -*-

from gensim.models import word2vec
import logging
import json

model = word2vec.Word2Vec.load("../data/output/words_v4.model")
res = model.most_similar("method")
# print json.dumps(res, encoding="utf-8", indent=3)

print model.most_similar('improve')
