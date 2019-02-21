# coding: utf-8
import numpy as np
import tensorflow as tf
import json
import random
from gensim.models import word2vec
from preprocessor import preprocessor
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
wordModel = word2vec.Word2Vec.load('./data/output/words_all.model')
f = open("./data/resource/label_map.json")
label_map = json.loads(f.read())
f.close()
label_list = []
for key in label_map:
    label_list.append(key)

LABEL_SIZE = len(label_list)
VECTOR_SIZE = 100
TRAIN_ITERS = 200
BATCH_SIZE = 20
HIDDEN_SIZE = 100
N_INPUTS = 100
LEARNING_RATE = 0.1
NUMBER_OF_LAYERS = 3
MAX_GRAD_NORM = 5                 # 用于控制梯度膨胀的梯度大小上限。
VOCAB_SIZE = 10000                # 词典规模。
IS_TRAIN = False                   # 控制程序是训练还是验证

class DataSet:
    X = []
    L = []
    Y = []


def cross_entropy(y, y_):
    return tf.nn.softmax_cross_entropy_with_logits(
        labels=y_,
        logits=y)


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
    res = np.zeros(LABEL_SIZE, dtype=float)
    i = 0
    for l in label_list:
        if label == l:
            res[i] = 1
            break
        i += 1
    return res


def read_data(path, split_rate, have_temp):
    # if have_temp:
    #     return read_temp()
    f = open(path, "r")
    issues = json.loads(f.read())
    f.close()
    random.shuffle(issues)
    test_data = DataSet()
    issue_length = len(issues)
    i = 0
    for issue in issues:
        issue_vec = text2vec(str(issue['title']) + ": " + issue['body'])
        label_vec = label2vec(issue['labels'])
        test_data.X.append(issue_vec)
        test_data.L.append(len(issue_vec))
        test_data.Y.append(label_vec)
        i += 1
        # if i == 40:
        #     return test_data
        logging.info('%d / %d...............%.4f%%' % (i, issue_length, i * 100.0 / issue_length))
    # make_temp(train_data,test_data)
    return test_data


def make_temp(train_data, test_data):
    f = open("./data/temp.train.X", "w")
    f.write(json.dumps(train_data.X, encoding="utf-8"))
    f.close()
    f = open("./data/temp.train.Y", "w")
    f.write(json.dumps(train_data.Y, encoding="utf-8"))
    f.close()
    f = open("./data/temp.train.L", "w")
    f.write(json.dumps(train_data.L, encoding="utf-8"))
    f.close()
    f = open("./data/temp.test.X", "w")
    f.write(json.dumps(test_data.X, encoding="utf-8"))
    f.close()
    f = open("./data/temp.test.Y", "w")
    f.write(json.dumps(test_data.Y, encoding="utf-8"))
    f.close()
    f = open("./data/temp.test.L", "w")
    f.write(json.dumps(test_data.L, encoding="utf-8"))
    f.close()


def read_temp():
    train_data = DataSet()
    test_data = DataSet()
    f = open("./data/temp.train.X", "r")
    train_data.X = json.loads(f.read())
    f.close()
    f = open("./data/temp.train.L", "r")
    train_data.L = json.loads(f.read())
    f.close()
    f = open("./data/temp.train.Y", "r")
    train_data.Y = json.loads(f.read())
    f.close()
    f = open("./data/temp.test.X", "r")
    test_data.X = json.loads(f.read())
    f.close()
    f = open("./data/temp.test.L", "r")
    test_data.L = json.loads(f.read())
    f.close()
    f = open("./data/temp.test.Y", "r")
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


def pre_recall_m(predict, answer):
    precision = []
    recall_ = []
    for tag in range(5):
        tp = count_tp(predict, answer, tag)
        fp = count_fp(predict, answer, tag)
        fn = count_fn(predict, answer, tag)
        if tp+fp > 0:
            precision.append(tp*1.0/(tp+fp))
        if tp+fn > 0:
            recall_.append(tp*1.0/(tp+fn))
    return precision, recall_


def pre_recall_u(predict, answer):
    tp_sum = 0
    tp_fp_sum = 0
    tp_fn_sum = 0
    precision = []
    recall_ = []
    for tag in range(5):
        tp = count_tp(predict, answer, tag)
        fp = count_fp(predict, answer, tag)
        fn = count_fn(predict, answer, tag)
        tp_sum += tp
        tp_fp_sum += tp+fp
        tp_fn_sum += tp+fn
    return tp_sum*1.0/tp_fp_sum, tp_sum*1.0/tp_fn_sum


def count_tp(predict, answer, tag):
    count = 0
    for i in range(len(predict)):
        if predict[i] == tag & answer[i] == tag:
            count += 1
    return count


def count_fp(predict, answer, tag):
    count = 0
    for i in range(len(predict)):
        if predict[i] == tag & answer[i] != tag:
            count += 1
    return count


def count_tn(predict, answer, tag):
    count = 0
    for i in range(len(predict)):
        if predict[i] != tag & answer[i] != tag:
            count += 1
    return count


def count_fn(predict, answer, tag):
    count = 0
    for i in range(len(predict)):
        if predict[i] != tag & answer[i] == tag:
            count += 1
    return count


graph = tf.get_default_graph()
test = read_data("./data/output/issue_corpus_2000_3000.ic", 0.7, False)
test_batches = make_batches(test, BATCH_SIZE)
with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./res_2000_3000/rnn-240.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./res_2000_3000'))
    # print(sess.run('x:0'))
    # print(sess.run('seq_len:0'))
    # print(sess.run('target_:0'))
    # print(sess.run('w:0'))
    # print(sess.run('b:0'))
    # print(sess.run('target:0'))
    x = graph.get_tensor_by_name('x:0')
    seq_len = graph.get_tensor_by_name('seq_len:0')
    target_ = graph.get_tensor_by_name('target_:0')
    w = graph.get_tensor_by_name('w:0')
    b = graph.get_tensor_by_name('b:0')
    target = tf.nn.softmax(graph.get_tensor_by_name('target:0'))
    predict = tf.argmax(target, 1)
    answer = tf.argmax(target_, 1)

    # 评估
    correct_prediction = tf.equal(tf.argmax(target, 1), tf.argmax(target_, 1))
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    temp = []
    temp_pre = []
    temp_answer = []
    for batch_x, batch_l, batch_y in test_batches:
        accuracy, cal_res, act_res, predict_res, answer_res = sess.run([accuracy_op, target, target_, predict, answer], feed_dict={x: batch_x, seq_len: batch_l, target_: batch_y})
        # print cal_res
        temp_pre += predict_res.tolist()
        temp_answer += answer_res.tolist()
        # print predict_res
        # print answer_res

        # print act_res
        temp.append(accuracy)
    # print(temp_pre)
    pre_m, recall_m = pre_recall_m(temp_pre, temp_answer)
    pre_u, recall_u = pre_recall_u(temp_pre, temp_answer)
    # print pre
    # print recall
    logging.info("Eval result:")
    logging.info("The avg accuracy is %f" % (np.mean(np.array(temp))))
    logging.info("The precision_m is %f" % np.mean(np.array((pre_m))))
    logging.info("The recall_m is %f" % np.mean(np.array(recall_m)))
    logging.info("The precision_u is %f" % pre_u)
    logging.info("The recall_u is %f" % recall_u)
    logging.info("The precision is")
    logging.info(pre_m)
    logging.info("The recall is ")
    logging.info(recall_m)
