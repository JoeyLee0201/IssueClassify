# -*- coding: UTF-8 -*-
import tensorflow as tf
import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 每个例子都有一个784位的向量，none代表例子的长度可以为任意长度
x = tf.placeholder("float", [None, 784])
# 权重值为784x10，即i=10, j = 784,初始值均为0
W = tf.Variable(tf.zeros([784, 10]))
# 偏置量的个数为10,初始值均为0
b = tf.Variable(tf.zeros([10]))


# tf.matmul(​​X，W)表示x乘以W
y = tf.nn.softmax(tf.matmul(x, W) + b)

# compute loss-the way of cross-entropy
y_ = tf.placeholder("float", [None, 10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# 设置训练参数，梯度下降每次为0.01
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 评估
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})