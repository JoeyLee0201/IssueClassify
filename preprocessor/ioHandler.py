# -*- coding: UTF-8 -*-

import preprocessor

def buildCorpus(originalFile, corpusFile):
    original = open(originalFile, "rb") 
    corpus = open(corpusFile, "w") 
    try:
        for line in original:
            if line: #不是空行
                sentences = preprocessor.preprocess(line.decode('utf-8'))
                for sentence in sentences:
                    if len(sentence):#不是空列表
                        for word in sentence:
                            corpus.write(word.encode('utf-8'))
                            corpus.write(" ")
                        corpus.write("\n")
            pass # do something
    except IOError,e:#检查open()是否失败，通常是IOError类型的错误
        print "***",e
    finally:
        original.close()
        corpus.close()
    return False

if __name__ == '__main__':
    print buildCorpus('../model.dat','../corpus.dat')