# -*- coding: utf-8 -*-  
from gensim import corpora, models, similarities, matutils
import logging  
from collections import defaultdict    
import preprocessor

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)  
  
#文档  
documents = ["Human machine interface for lab abc computer applications",  
"A survey of user opinion of computer system response time",   
"The EPS user interface management system",   
"System and human system engineering testing of EPS",  
"Relation of user perceived response time to error measurement",  
"The generation of random binary unordered trees",    
"The intersection graph of paths in trees",    
"Graph minors IV Widths of trees and well quasi ordering",    
"Graph minors A survey"]  
  
#1.分词，去除停用词  
texts=[]
for document in documents:
    texts = texts+preprocessor.preprocess(document.decode('utf-8'))
  
#2.计算词频  
frequency = defaultdict(int) #构建一个字典对象  
#遍历分词后的结果集，计算每个词出现的频率  
for text in texts:  
    for token in text:  
        frequency[token]+=1  
#选择频率大于1的词  
texts=[[token for token in text if frequency[token]>1] for text in texts] 
  
#3.创建字典（单词与编号之间的映射）  
dictionary=corpora.Dictionary(texts)
  
#4.将要比较的文档转换为向量（词袋表示方法）  
#要比较的文档  
new_doc = "Human computer interaction"  
#将文档分词并使用doc2bow方法对每个不同单词的词频进行了统计，并将单词转换为其编号，然后以稀疏向量的形式返回结果  
new_vec = dictionary.doc2bow(new_doc.lower().split()) 

#5.建立语料库  
#将每一篇文档转换为向量  
corpus = [dictionary.doc2bow(text) for text in texts] 

#6.初始化模型  
# 初始化一个tfidf模型,可以用它来转换向量（词袋整数计数）表示方法为新的表示方法（Tfidf 实数权重）  
tfidf = models.TfidfModel(corpus)  
#将整个语料库转为tfidf表示方法  
corpus_tfidf = tfidf[corpus]  

#7.创建索引  
index = similarities.MatrixSimilarity(corpus_tfidf)  

#8.相似度计算  
new_vec_tfidf=tfidf[new_vec]#将要比较文档转换为tfidf表示方法  
#计算要比较的文档与语料库中每篇文档的相似度  
sims = index[new_vec_tfidf]  
# print(list(enumerate(sims)))
print(sorted(enumerate(sims), key=lambda item: -item[1]))
#[ 0.81649655  0.31412902  0.          0.34777319  0.          0.          0.  
#  0.          0.        ] 
print new_vec_tfidf
for t in corpus_tfidf:
    print matutils.cossim(new_vec_tfidf,t)