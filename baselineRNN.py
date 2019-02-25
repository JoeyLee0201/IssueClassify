# -*- coding: UTF-8 -*-


from gensim.models import word2vec
from preprocessor import preprocessor
from gensim.models.doc2vec import Doc2Vec

import tensorflow as tf
import numpy as np
import logging
import gc
import os
import json
import random


class DataSet:
    X = []
    L = []
    Y = []


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
tf.get_default_graph()
wordModel = word2vec.Word2Vec.load('./data/output/words_baseline.model')


LABEL_SIZE = 13
VECTOR_SIZE = 100
TRAIN_ITERS = 300
BATCH_SIZE = 20
HIDDEN_SIZE = 100
N_INPUTS = 100
LEARNING_RATE = 0.01
NUMBER_OF_LAYERS = 3
MAX_GRAD_NORM = 5                 # 用于控制梯度膨胀的梯度大小上限。
VOCAB_SIZE = 10000                # 词典规模。
IS_TRAIN = True                   # 控制程序是训练还是验证
IS_TEST = False
LABELS = ["BUG", "RFE", "IMPROVEMENT", "DOCUMENTATION", "TASK", "BUILD SYSTEM", "REFACTORING", "DESIGN DEFECT",
          "TEST", "CLEANUP", "BACKPORT", "SPECIFICATION", "OTHERS"]


def cross_entropy(y, y_):
    return tf.nn.softmax_cross_entropy_with_logits(
        labels=y_,
        logits=y)
    # return -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
    # return -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-8, tf.reduce_max(y))))


# text data
def text2vec(text):
    words = preprocessor.preprocessToWord(text)
    res = []
    for word in words:
        try:
            res.append(wordModel[word])
        except KeyError:
            res.append(np.zeros(VECTOR_SIZE))
    return res


def label2vec(label):
    if label == "UNKNOWN":
        label = "OTHERS"
    res = np.zeros(LABEL_SIZE, dtype=float)
    i = 0
    for l in LABELS:
        if label == l:
            res[i] = 1
            break
        i += 1
    return res
    # if label == 'bug':
    #     return np.array([1, 0, 0])
    # elif label == 'enhancement':
    #     return np.array([0, 1, 0])
    # else:
    #     return np.array([0, 0, 1])


def read_data(path, split_rate, have_temp=False, labels=None):
    # if have_temp:
    #     return read_temp()
    train_data = DataSet()
    test_data = DataSet()
    issues = []
    if labels is None:
        f = open(path, "r")
        issues = json.loads(f.read())
        f.close()
        random.shuffle(issues)
    else:
        for label in labels:
            f = open(path+label+".ic", "r")
            issues += json.loads(f.read())
            f.close()
        random.shuffle(issues)

    issue_length = len(issues)
    i = 0
    if IS_TRAIN and not IS_TEST:
        point = split_rate * issue_length
    else:
        point = 140
    for issue in issues:
        issue_vec = text2vec(unicode(issue['title']) + ": " + unicode(issue['body']))
        label_vec = label2vec(issue['labels'])
        if i <= point:
            train_data.X.append(issue_vec)
            train_data.L.append(len(issue_vec))
            train_data.Y.append(label_vec)
        else:
            test_data.X.append(issue_vec)
            test_data.L.append(len(issue_vec))
            test_data.Y.append(label_vec)
        i += 1
        if i == 200 and (IS_TEST or not IS_TRAIN):
            # make_temp(train_data, test_data)
            return train_data, test_data
        logging.info('%d / %d...............%.4f%%' % (i, issue_length, i * 100.0 / issue_length))
    # make_temp(train_data, test_data)

    return train_data, test_data


def make_temp(train_data, test_data):
    f = open("./data/train.X", "w")
    f.write(json.dumps(train_data.X, encoding="utf-8"))
    f.close()
    f = open("./data/train.Y", "w")
    f.write(json.dumps(train_data.Y, encoding="utf-8"))
    f.close()
    f = open("./data/train.L", "w")
    f.write(json.dumps(train_data.L, encoding="utf-8"))
    f.close()
    f = open("./data/test.X", "w")
    f.write(json.dumps(test_data.X, encoding="utf-8"))
    f.close()
    f = open("./data/test.Y", "w")
    f.write(json.dumps(test_data.Y, encoding="utf-8"))
    f.close()
    f = open("./data/test.L", "w")
    f.write(json.dumps(test_data.L, encoding="utf-8"))
    f.close()


def read_temp():
    train_data = DataSet()
    test_data = DataSet()
    f = open("./data/train.X", "r")
    train_data.X = json.loads(f.read())
    f.close()
    f = open("./data/train.L", "r")
    train_data.L = json.loads(f.read())
    f.close()
    f = open("./data/train.Y", "r")
    train_data.Y = json.loads(f.read())
    f.close()
    f = open("./data/test.X", "r")
    test_data.X = json.loads(f.read())
    f.close()
    f = open("./data/test.L", "r")
    test_data.L = json.loads(f.read())
    f.close()
    f = open("./data/test.Y", "r")
    test_data.Y = json.loads(f.read())
    f.close()
    return train_data, test_data


def make_batches(input_data, batch_size):
    num_batches = len(input_data.Y) // batch_size
    data = np.array(input_data.X[: batch_size*num_batches])
    data = np.reshape(data, [batch_size, num_batches])
    data_batches = np.split(data, num_batches, axis=1)
    data_batches_rs = []
    for one_batch in data_batches:
        sub_batch = []
        max_seq_len = 0
        for one_issue in one_batch:
            max_seq_len = max(max_seq_len, len(one_issue[0]))
        for one_issue in one_batch:
            todo = max_seq_len - len(one_issue[0])
            for index in range(todo):
                one_issue[0].append(np.zeros(VECTOR_SIZE))
            sub_batch.append(np.array(one_issue[0]))
        data_batches_rs.append(np.array(sub_batch))

    length = np.array(input_data.L[: batch_size * num_batches])
    length = np.reshape(length, [batch_size, num_batches])
    len_batches = np.split(length, num_batches, axis=1)
    len_batches = np.reshape(np.array(len_batches), [num_batches, BATCH_SIZE])

    label = np.array(input_data.Y[: batch_size * num_batches])
    label = np.reshape(label, [batch_size, num_batches, LABEL_SIZE])
    label_batches = np.split(label, num_batches, axis=1)
    label_batches_rs = []
    for one_batch in label_batches:
        sub_batch = []
        for one_label in one_batch:
            sub_batch.append(one_label[0])
        label_batches_rs.append(np.array(sub_batch))

    print 'make batches done'
    return list(zip(data_batches_rs, len_batches, label_batches_rs))


def rnn_model(input_data, seq_len):
    rnn_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(NUMBER_OF_LAYERS)])
    outputs_, state = tf.nn.dynamic_rnn(rnn_cell, input_data, sequence_length=seq_len, dtype=tf.float32)
    return outputs_, state


x = tf.placeholder(tf.float32, [BATCH_SIZE, None, VECTOR_SIZE], name="x")
seq_len = tf.placeholder(tf.int32, [BATCH_SIZE, ], name="seq_len")

target_ = tf.placeholder(tf.float32, [BATCH_SIZE, LABEL_SIZE], name="target_")

with tf.variable_scope("issue", reuse=tf.AUTO_REUSE):
    outputs, states = rnn_model(x, seq_len)

# output = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])
output = states[-1].h
w = tf.Variable(tf.zeros([HIDDEN_SIZE, LABEL_SIZE]), name="w")
b = tf.Variable(tf.zeros([LABEL_SIZE]), name="b")
logits = tf.matmul(output, w) + b
target = tf.nn.softmax(logits, name="target")
#
# # 定义交叉熵损失函数和平均损失。
# loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
#             labels=tf.reshape(target, [-1]),
#             logits=logits)
# cost = tf.reduce_sum(loss) / BATCH_SIZE
loss_op = cross_entropy(logits, target_)
# loss_op = cross_entropy(target, target_)
# loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(
#     labels=target_,
#     logits=logits
# )
optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
train_op = optimizer.minimize(loss_op)

# 评估
correct_prediction = tf.equal(tf.argmax(target, 1), tf.argmax(target_, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, "float"))

init = tf.global_variables_initializer()
train, test = read_data("./data/output/baseline_issue_corpus_for_train_1.ic", 0.7, have_temp=False, labels=None)
train_batches = make_batches(train, BATCH_SIZE)
test_batches = make_batches(test, BATCH_SIZE)
split = len(test_batches)*2/3
total_tests = len(test_batches) * BATCH_SIZE

if IS_TRAIN:
    with tf.Session() as sess:
        saver = tf.train.Saver(max_to_keep=1)
        max_acc = 0
        sess.run(init)

        for step in range(TRAIN_ITERS):
            logging.info("Step: " + str(step))
            # temp1 = []
            # temp2 = []
            for batch_x, batch_l, batch_y in train_batches:
                loss, _, accuracy = sess.run([loss_op, train_op, accuracy_op],
                                             feed_dict={x: batch_x, seq_len: batch_l, target_: batch_y})
                # temp1.append(loss)
                # temp2.append(accuracy)
            # if step % 10 == 0:
            #     # print("loss: ", str(temp1))
            #     # print("accutacy: ", str(temp2))
            #     logging.info("Training result:")
            #     logging.info("At the step %d, the avg loss is %f, the accuracy is %f" % (step, np.mean(np.array(temp1)),
            #                                                                              np.mean(np.array(temp2))))

            if step % 10 == 0:
                temp3 = []
                temp4 = []
                # i = 0
                for batch_x2, batch_l2, batch_y2 in test_batches:
                    # logging.info(i)
                    # if i == split:
                    #     break
                    # i += 1
                    loss, accuracy, cal_res, act_res = sess.run([loss_op, accuracy_op, target, target_],
                                              feed_dict={x: batch_x2, seq_len: batch_l2, target_: batch_y2})
                    # print loss
                    temp3.append(loss)
                    temp4.append(accuracy)
                    # if IS_TEST:
                    #     print "train result:"
                    #     print cal_res
                    #     print "actual result:"
                    #     print act_res
                # logging.info(str(temp3))
                # logging.info(str(temp4))
                temp_loss = np.mean(np.array(temp3))
                temp_acc = np.mean(np.array(temp4))
                logging.info("Testing result:")
                logging.info("At the step %d, the avg loss is %f, the accuracy is %f" % (step, temp_loss, temp_acc))
                if temp_acc > max_acc:
                    max_acc = temp_acc
                    saver.save(sess, './rnn', global_step=step)
        temp3 = []
        temp4 = []
        # i = split
        for batch_x2, batch_l2, batch_y2 in test_batches:
            # if i < split:
            #     i += 1
            #     continue
            # logging.info(i)
            # i += 1
            loss, accuracy = sess.run([loss_op, accuracy_op],
                                      feed_dict={x: batch_x2, seq_len: batch_l2, target_: batch_y2})
            # print loss
            temp3.append(loss)
            temp4.append(accuracy)
        # logging.info(str(temp3))
        # logging.info(str(temp4))
        temp_loss = np.mean(np.array(temp3))
        temp_acc = np.mean(np.array(temp4))
        logging.info("Finally result:")
        logging.info("The avg loss is %f, the accuracy is %f" % (temp_loss, temp_acc))
        if temp_acc > max_acc:
            max_acc = temp_acc
            saver.save(sess, './rnn', global_step=TRAIN_ITERS)
        logging.info("Optimization Finished!")
else:
    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.import_meta_graph('./res2/rnn-200.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./res2'))
        print(sess.run('b:0'))
        # temp1 = []
        # temp2 = []
        # for batch_x, batch_l, batch_y in test_batches:
        #     loss, accuracy = sess.run([loss_op, accuracy_op],
        #                               feed_dict={x: batch_x, seq_len: batch_l,
        #                                          target_: batch_y})
        #     # print loss
        #     temp1.append(loss)
        #     temp2.append(accuracy)
        # logging.info("Testing result:")
        # logging.info("The avg loss is %f, the accuracy is %f" % (np.mean(np.array(temp1)),
        #                                                          np.mean(np.array(temp2))))



# lstm = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
# # # 初始化 LSTM 存储状态.
# state = lstm.zero_state(BATCH_SIZE, tf.float32)
# #
# loss = 0.0
# for current_batch_of_issue, current_batch_of_size, current_batch_of_label in train_batches:
#     # 每次处理一批词语后更新状态值.
#     output, state = lstm(current_batch_of_issue, state)
#
#     # LSTM 输出可用于产生下一个词语的预测
#     # logits = tf.matmul(output, w) + b
#     logits = tf.nn.bias_add(tf.matmul(output, w), b)
#     probabilities = tf.nn.softmax(logits)
#     loss += cross_entropy(probabilities, target_)
#
# print loss


########################################################################
# def LSTM_sample():
#     lstm = rnn.cell.BasicLSTMCell(lstm_size)
#     # 初始化 LSTM 存储状态.
#     state = tf.zeros([batch_size, lstm.state_size])
#
#     loss = 0.0
#     for current_batch_of_words in words_in_dataset:
#         # 每次处理一批词语后更新状态值.
#         output, state = lstm(current_batch_of_words, state)
#
#         # LSTM 输出可用于产生下一个词语的预测
#         logits = tf.matmul(output, softmax_w) + softmax_b
#         probabilities = tf.nn.softmax(logits)
#         loss += loss_function(probabilities, target_words)


# def break_sample():
#     # 一次给定的迭代中的输入占位符.
#     words = tf.placeholder(tf.int32, [batch_size, num_steps])
#
#     lstm = rnn_cell.BasicLSTMCell(lstm_size)
#     # 初始化 LSTM 存储状态.
#     initial_state = state = tf.zeros([batch_size, lstm.state_size])
#
#     for i in range(len(num_steps)):
#         # 每处理一批词语后更新状态值.
#         output, state = lstm(words[:, i], state)
#
#         # 其余的代码.
#         # ...
#
#     final_state = state


# def iterat_sample():
#     # 一个 numpy 数组，保存每一批词语之后的 LSTM 状态.
#     numpy_state = initial_state.eval()
#     total_loss = 0.0
#     for current_batch_of_words in words_in_dataset:
#         numpy_state, current_loss = session.run([final_state, loss],
#               # 通过上一次迭代结果初始化 LSTM 状态.
#               feed_dict={initial_state: numpy_state, words: current_batch_of_words})
#         total_loss += current_loss


# def multi_LSTM_sample():
#     lstm = rnn_cell.BasicLSTMCell(lstm_size)
#     stacked_lstm = rnn_cell.MultiRNNCell([lstm] * number_of_layers)
#
#     initial_state = state = stacked_lstm.zero_state(batch_size, tf.float32)
#     for i in range(len(num_steps)):
#         # 每次处理一批词语后更新状态值.
#         output, state = stacked_lstm(words[:, i], state)
#
#         # 其余的代码.
#         # ...
#
#     final_state = state



