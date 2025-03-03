# coding: utf-8
import numpy as np
import tensorflow as tf
import json
import random
from gensim.models import word2vec
from preprocessor import preprocessor

TRAIN_DATA = "./data/train_data"  # 训练数据路径。
EVAL_DATA = "./data/valid_data"   # 验证数据路径。
TEST_DATA = "./data/test_data"    # 测试数据路径。
HIDDEN_SIZE = 300                 # 隐藏层规模。
NUM_LAYERS = 2                    # 深层循环神经网络中LSTM结构的层数。
VOCAB_SIZE = 10000                # 词典规模。
VECTOR_SIZE = 100                 # 词向量维度
LABEL_SIZE = 3                    # 标签规模
TRAIN_BATCH_SIZE = 20             # 训练数据batch的大小。
# TRAIN_NUM_STEP = 35               # 训练数据截断长度。
TRAIN_ITERS = 1000                # 训练的迭代次数

EVAL_BATCH_SIZE = 1               # 测试数据batch的大小。
EVAL_NUM_STEP = 1                 # 测试数据截断长度。
NUM_EPOCH = 5                     # 使用训练数据的轮数。
LSTM_KEEP_PROB = 0.9              # LSTM节点不被dropout的概率。
EMBEDDING_KEEP_PROB = 0.9         # 词向量不被dropout的概率。
MAX_GRAD_NORM = 5                 # 用于控制梯度膨胀的梯度大小上限。
SHARE_EMB_AND_SOFTMAX = True      # 在Softmax层和词向量层之间共享参数。

EMBEDDING_MODEL = word2vec.Word2Vec.load('./data/output/words2.model')


class DataSet:
    X = []
    L = []
    Y = []


# 通过一个RNNModel类来描述模型，这样方便维护循环神经网络中的状态。
class RNNModel(object):
    def __init__(self, is_training, batch_size, vector_size, label_size):
        # 记录使用的batch大小和截断长度。
        self.batch_size = batch_size
        self.vector_size = vector_size
        self.label_size = label_size

        # 定义每一步的输入和预期输出。
        # 输入是[batch_size, None, VECTOR_SIZE]，每批的句子长度（单词数量）不等，每个单词维度都是相同的；
        # 由于不定长，需要指定每个样本的长度；
        # 输出是[batch_size, LABEL_SIZE]，LABEL_SIZE决定每个样本可打的标签数量，目前是3个标签
        self.input_data = tf.placeholder(tf.float32, [batch_size, None, vector_size])
        self.seq_len = tf.placeholder(tf.float32, [batch_size, ])
        self.targets = tf.placeholder(tf.float32, [batch_size, 1, label_size])

        # 定义使用LSTM结构为循环体结构且使用dropout的深层循环神经网络;
        dropout_keep_prob = LSTM_KEEP_PROB if is_training else 1.0
        lstm_cells = [
            tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE),
                output_keep_prob=dropout_keep_prob)
            for _ in range(NUM_LAYERS)]
        cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)

        # 初始化最初的状态，即全零的向量。这个量只在每个epoch初始化第一个batch
        # 时使用。
        self.initial_state = cell.zero_state(batch_size, tf.float32)

        ###################################################################
        # 因为有自制embedding层，input_data已经是词向量#
        # 定义单词的词向量矩阵。
        # embedding = tf.get_variable("embedding", [VOCAB_SIZE, HIDDEN_SIZE])

        # 将输入单词转化为词向量。
        # inputs = tf.nn.embedding_lookup(embedding, self.input_data)
        inputs = self.input_data
        ###################################################################
        # 只在训练时使用dropout。
        if is_training:
            inputs = tf.nn.dropout(inputs, EMBEDDING_KEEP_PROB)

        # 定义输出列表。在这里先将不同时刻LSTM结构的输出收集起来，再一起提供给
        # softmax层。
        outputs = []
        state = self.initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(TRAIN_ITERS):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                ###########################################################
                # 此处应该修改为动态指定长度#
                cell_output, state = tf.nn.dynamic_rnn(cell, inputs,
                                                       sequence_length=self.seq_len, dtype=tf.float32)
                # cell_output, state = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)
                # 把输出队列展开成[batch, hidden_size*num_steps]的形状，然后再
        # reshape成[batch*numsteps, hidden_size]的形状。
        output = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])
        ##################################################################

        # Softmax层：将RNN在每个位置上的输出转化为各个单词的logits。
        if SHARE_EMB_AND_SOFTMAX:
            weight = tf.transpose(EMBEDDING_MODEL)
        else:
            weight = tf.get_variable("weight", [HIDDEN_SIZE, VOCAB_SIZE])
        bias = tf.get_variable("bias", [VOCAB_SIZE])
        logits = tf.matmul(output, weight) + bias

        # 定义交叉熵损失函数和平均损失。
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(self.targets, [-1]),
            logits=logits)
        self.cost = tf.reduce_sum(loss) / batch_size
        self.final_state = state

        # 只在训练模型时定义反向传播操作。
        if not is_training:
            return

        trainable_variables = tf.trainable_variables()
        # 控制梯度大小，定义优化方法和训练步骤。
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.cost, trainable_variables), MAX_GRAD_NORM)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        self.train_op = optimizer.apply_gradients(
            zip(grads, trainable_variables))


# 使用给定的模型model在数据data上运行train_op并返回在全部数据上的perplexity值。
def run_epoch(session, model, batches, train_op, output_log, step):
    # 计算平均perplexity的辅助变量。
    total_costs = 0.0
    iters = 0
    state = session.run(model.initial_state)
    # 训练一个epoch。
    for x, l, y in batches:
        # 在当前batch上运行train_op并计算损失值。交叉熵损失函数计算的就是下一个单
        # 词为给定单词的概率。
        cost, state, _ = session.run(
             [model.cost, model.final_state, train_op],
             {model.input_data: x, model.seq_len: l, model.targets: y,
              model.initial_state: state})
        total_costs += cost
        iters += model.num_steps

        # 只有在训练时输出日志。
        if output_log and step % 100 == 0:
            print("After %d steps, perplexity is %.3f" % (
                  step, np.exp(total_costs / iters)))
        step += 1

    # 返回给定模型在给定数据上的perplexity值。
    return step, np.exp(total_costs / iters)


def text2vec(text):
    words = preprocessor.preprocessToWord(text)
    res = []
    for word in words:
        try:
            res.append(EMBEDDING_MODEL[word])
        except KeyError:
            res.append(np.zeros(VECTOR_SIZE))
    return res


def label2vec(label):
    if label == 'bug':
        return np.array([1, 0, 0])
    if label == 'enhancement':
        return np.array([0, 1, 0])
    if label == 'documentation':
        return np.array([0, 0, 1])


# 从文件中读取数据，并返回包含单词编号的数组。
def read_data(path, split_rate):
    # with open(file_path, "r") as fin:
    #     # 将整个文档读进一个长字符串。
    #     id_string = ' '.join([line.strip() for line in fin.readlines()])
    # id_list = [int(w) for w in id_string.split()]  # 将读取的单词编号转为整数
    # return id_list
    # 目前数据没有进行切分，暂时手动分割训练和验证集
    f = open(path, "r")
    issues = json.loads(f.read())
    f.close()
    random.shuffle(issues)
    train_data = DataSet()
    test_data = DataSet()
    issue_length = len(issues)
    i = 0
    point = split_rate * issue_length
    for issue in issues:
        issue_vec = text2vec(issue['title'] + ": " + issue['body'])
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
    return train_data, test_data


# def make_batches(id_list, batch_size, num_step):
#     # 计算总的batch数量。每个batch包含的单词数量是batch_size * num_step。
#     num_batches = (len(id_list) - 1) // (batch_size * num_step)
#
#     # 如9-4图所示，将数据整理成一个维度为[batch_size, num_batches * num_step]
#     # 的二维数组。
#     data = np.array(id_list[: num_batches * batch_size * num_step])
#     data = np.reshape(data, [batch_size, num_batches * num_step])
#     # 沿着第二个维度将数据切分成num_batches个batch，存入一个数组。
#     data_batches = np.split(data, num_batches, axis=1)
#
#     # 重复上述操作，但是每个位置向右移动一位。这里得到的是RNN每一步输出所需要预测的
#     # 下一个单词。
#     label = np.array(id_list[1 : num_batches * batch_size * num_step + 1])
#     label = np.reshape(label, [batch_size, num_batches * num_step])
#     label_batches = np.split(label, num_batches, axis=1)
#     # 返回一个长度为num_batches的数组，其中每一项包括一个data矩阵和一个label矩阵。
#     return list(zip(data_batches, label_batches))
def make_batches(input_data, batch_size):
    num_batches = len(input_data.Y) // batch_size
    data = np.array(input_data.X[: batch_size * num_batches])
    data = np.reshape(data, [batch_size, num_batches])
    data_batches = np.split(data, num_batches, axis=1)
    data_batches_rs = []
    del data
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
    label = np.reshape(label, [batch_size, num_batches, 3])
    label_batches = np.split(label, num_batches, axis=1)
    print 'make batches done'
    return list(zip(data_batches_rs, len_batches, label_batches))


def main():
    print 'start'
    # 定义初始化函数。
    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    print 'defining rnn model for training....'
    # 定义训练用的循环神经网络模型。
    with tf.variable_scope("language_model",
                           reuse=None, initializer=initializer):
        train_model = RNNModel(True, TRAIN_BATCH_SIZE, VECTOR_SIZE, LABEL_SIZE)
    print 'defining rnn model for testing....'
    # 定义测试用的循环神经网络模型。它与train_model共用参数，但是没有dropout。
    with tf.variable_scope("language_model",
                           reuse=True, initializer=initializer):
        eval_model = RNNModel(False, EVAL_BATCH_SIZE, VECTOR_SIZE, LABEL_SIZE)
    print 'training....'
    # 训练模型。
    with tf.Session() as session:
        tf.global_variables_initializer().run()
        print '     load data...'
        train_data, eval_data = read_data("./resource/issue_corpus_random.ic", 0.7)
        print '     make batch...'
        train_batches = make_batches(train_data, TRAIN_BATCH_SIZE)
        eval_batches = make_batches(eval_data, EVAL_BATCH_SIZE)
        # train_batches = make_batches(
        #     read_data(TRAIN_DATA), TRAIN_BATCH_SIZE, VECTOR_SIZE, LABEL_SIZE)
        # eval_batches = make_batches(
        #     read_data(EVAL_DATA), EVAL_BATCH_SIZE, EVAL_NUM_STEP)
        # test_batches = make_batches(
        #     read_data(TEST_DATA), EVAL_BATCH_SIZE, EVAL_NUM_STEP)
        print ''
        print ''
        step = 0
        for i in range(NUM_EPOCH):
            print("In iteration: %d" % (i + 1))
            step, train_pplx = run_epoch(session, train_model, train_batches,
                                         train_model.train_op, True, step)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_pplx))

            _, eval_pplx = run_epoch(session, eval_model, eval_batches,
                                     tf.no_op(), False, 0)
        #     print("Epoch: %d Eval Perplexity: %.3f" % (i + 1, eval_pplx))
        #
        # _, test_pplx = run_epoch(session, eval_model, test_batches,
        #                          tf.no_op(), False, 0)
        # print("Test Perplexity: %.3f" % test_pplx)


if __name__ == "__main__":
    main()
