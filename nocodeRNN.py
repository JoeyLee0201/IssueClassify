# -*- coding: UTF-8 -*-
from gensim.models import word2vec
from preprocessor import preprocessor

import tensorflow as tf
import numpy as np
import logging
import os
import json

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

VECTOR_SIZE = 100
TRAIN_ITERS = 1000
BATCH_SIZE = 20
HIDDEN_SIZE = 100
N_INPUTS = 100
LEARNING_RATE = 0.001

wordModel = word2vec.Word2Vec.load('test/nocode50904245.model')


# text data
def text2vec(text, isHtml):
    if isHtml:
        words = preprocessor.processHTML(text)[1]
    else:
        words = preprocessor.preprocessToWord(text)
    res = []
    for word in words:
        try:
            res.append(wordModel[word])
        except KeyError:
            res.append(np.zeros(VECTOR_SIZE))
    return res


#  shape = [None, seq len, Vec size]
def read_data(path='./train'):
    X1 = []
    X2 = []
    L1 = []
    L2 = []
    Y = []
    filelist = os.listdir(path)
    for i in range(0, len(filelist)):
    # for i in range(0, 1):
        filepath = os.path.join(path, filelist[i])
        logging.info("Loaded the file:"+filepath)
        if os.path.isfile(filepath):
            file = open(filepath, 'rb')
            testlist = json.loads(file.read())
            for map in testlist:
                commit = text2vec(map['commit'], False)
                issue = text2vec(map['issue'], True)
                L1.append(len(commit))
                X1.append(commit)
                L2.append(len(issue))
                X2.append(issue)
                Y.append(float(map['type']))
            file.close()
    return X1, X2, L1, L2, Y


# shape=[batch_size, None]
def make_batches(data, batch_size):
    X1, X2, L1, L2, Y = data
    num_batches = len(Y) // batch_size
    data1 = np.array(X1[: batch_size*num_batches])
    data1 = np.reshape(data1, [batch_size, num_batches])
    data_batches1 = np.split(data1, num_batches, axis=1)  #  list
    data_batches1_rs = []
    for d1 in data_batches1:
        sub_batch = []
        maxD = 0
        for d in d1:
            for dt in d:
                maxD = max(maxD, len(dt))
        for d in d1:
            for dt in d:
                todo = maxD - len(dt)
                for index in range(todo):
                    dt.append(np.zeros(VECTOR_SIZE))
                sub_batch.append(np.array(dt))
        data_batches1_rs.append(np.array(sub_batch))

    data2 = np.array(X2[: batch_size*num_batches])
    data2 = np.reshape(data2, [batch_size, num_batches])
    data_batches2 = np.split(data2, num_batches, axis=1)
    data_batches2_rs = []
    for d2 in data_batches2:
        sub_batch = []
        maxD = 0
        for d in d2:
            for dt in d:
                maxD = max(maxD, len(dt))
        for d in d2:
            for dt in d:
                todo = maxD - len(dt)
                for index in range(todo):
                    dt.append(np.zeros(VECTOR_SIZE))
                sub_batch.append(np.array(dt))
        data_batches2_rs.append(np.array(sub_batch))

    len1 = np.array(L1[: batch_size*num_batches])
    len1 = np.reshape(len1, [batch_size, num_batches])
    len_batches1 = np.split(len1, num_batches, axis=1)
    len_batches1 = np.reshape(np.array(len_batches1), [num_batches, BATCH_SIZE])

    len2 = np.array(L2[: batch_size * num_batches])
    len2 = np.reshape(len2, [batch_size, num_batches])
    len_batches2 = np.split(len2, num_batches, axis=1)
    len_batches2 = np.reshape(np.array(len_batches2), [num_batches, BATCH_SIZE])

    label = np.array(Y[: batch_size*num_batches])
    label = np.reshape(label, [batch_size, num_batches])
    label_batches = np.split(label, num_batches, axis=1)
    return list(zip(data_batches1_rs, data_batches2_rs, len_batches1, len_batches2, label_batches))


input1 = tf.placeholder(tf.float64, [BATCH_SIZE, None, VECTOR_SIZE])
input2 = tf.placeholder(tf.float64, [BATCH_SIZE, None, VECTOR_SIZE])
len1 = tf.placeholder(tf.int64, [BATCH_SIZE, ])
len2 = tf.placeholder(tf.int64, [BATCH_SIZE, ])
target = tf.placeholder(tf.float64, [BATCH_SIZE, 1])


def RNN(input_data, seq_len):
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
    rnn_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell for _ in range(3)])
    outputs, state = tf.nn.dynamic_rnn(rnn_cell, input_data, sequence_length=seq_len, dtype=tf.float64)
    return outputs, state


# initializer = tf.random_uniform_initializer(-0.5, 0.5, dtype=tf.float32)
with tf.variable_scope("commit", reuse=tf.AUTO_REUSE):
    outputs1, states1 = RNN(input1, len1)
with tf.variable_scope("issue", reuse=tf.AUTO_REUSE):
    outputs2, states2 = RNN(input2, len2)

newoutput1 = states1[-1].h
newoutput2 = states2[-1].h


def getScore(state1, state2):
    pooled_len_1 = tf.sqrt(tf.reduce_sum(state1 * state1, 1))
    pooled_len_2 = tf.sqrt(tf.reduce_sum(state2 * state2, 1))
    pooled_mul_12 = tf.reduce_sum(state1 * state2, 1)
    score = tf.div(pooled_mul_12, pooled_len_1 * pooled_len_2 + 1e-8, name="scores")  # +1e-8 avoid 'len_1/len_2 == 0'
    score = tf.reshape(score, [BATCH_SIZE, 1])
    return score

#  |t - cossimilar(state1, state2)|
def getLoss(score, t):
    # pooled_len_1 = tf.sqrt(tf.reduce_sum(state1 * state1, 1))
    # pooled_len_2 = tf.sqrt(tf.reduce_sum(state2 * state2, 1))
    # pooled_mul_12 = tf.reduce_sum(state1 * state2, 1)
    # score = tf.div(pooled_mul_12, pooled_len_1 * pooled_len_2+1e-8, name="scores")  #  +1e-8 avoid 'len_1/len_2 == 0'
    # score = tf.reshape(score, [BATCH_SIZE, 1])
    rs = t - score
    rs = tf.abs(rs)
    return tf.reduce_mean(rs)


# Define loss and optimizer
cos_score = getScore(newoutput1, newoutput2)
loss_op = getLoss(cos_score, target)

optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train_op = optimizer.minimize(loss_op)


def get_correct(score, target):
    rs = target - score
    rs = np.abs(rs)
    result = 0
    for onescore in rs:
        if onescore[0] < 0.5:
            result = result + 1
    return result


# writer = tf.summary.FileWriter('log/graphlog', tf.get_default_graph())
# writer.close()
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
train_batches = make_batches(read_data(), BATCH_SIZE)
test_batches = make_batches(read_data(path="./testset"), BATCH_SIZE)
total_tests = len(test_batches) * BATCH_SIZE
with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(init)

    for step in range(TRAIN_ITERS):
        logging.info("Step: " + str(step))
        for x1, x2, l1, l2, y in train_batches:
            loss, _ = sess.run([loss_op, train_op], feed_dict={input1: x1, input2: x2, len1: l1, len2: l2, target: y})

        if step % 100 == 0:
            temp = []
            total_correct = 0
            for x1, x2, l1, l2, y in test_batches:
                score, loss = sess.run([cos_score, loss_op], feed_dict={input1: x1, input2: x2, len1: l1, len2: l2, target: y})
                temp.append(loss)
                total_correct = total_correct + get_correct(score, y)
            logging.info(str(temp))
            logging.info("At the step %d, the avg loss is %f, the accuracy is %f" % (step, np.mean(np.array(temp)), float(total_correct)/total_tests))
    saver.save(sess, 'rnnmodel/adam/rnn', global_step=TRAIN_ITERS)
    logging.info("Optimization Finished!")