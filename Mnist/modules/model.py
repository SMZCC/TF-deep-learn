# coding=utf-8
# date: 2018/12/11, 16:34
# name: smz


import tensorflow as tf
from Mnist.configrations.options import *

class MnistSoftmax(object):
    def __init__(self):
        tf.reset_default_graph()
        self.__writer__ = None
        self.__saver__ = None

        self.input = tf.placeholder(dtype=tf.float32, shape=[None, 784], name="Input")
        self.label = tf.placeholder(dtype=tf.float32, shape=[None, 10], name="Label")
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        tf.summary.image("input0", tf.reshape(self.input, [BATCH_SIZE, 28, 28, 1]), max_outputs=BATCH_SIZE)    # tf.summary.image()收集的数据必须是一个4-D的张量

    def build(self, logs_dir):
        with tf.variable_scope("params") as param:
            self.weights = tf.get_variable(name="weights", shape=[784, 10], initializer=tf.random_normal_initializer())
            self.biases = tf.get_variable(name="biases", shape=[10], initializer=tf.constant_initializer(0.1))

        self.logits = tf.matmul(self.input, self.weights) + self.biases
        self.pred = tf.nn.softmax(self.logits)   # 使用softmax来预测类别
        self.loss = tf.reduce_mean(-tf.reduce_sum(self.label * tf.log(self.pred), axis=1))
        tf.summary.scalar("loss", self.loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        self.train_step = self.optimizer.minimize(self.loss, global_step=self.global_step)

        self.merge_op = tf.summary.merge_all()
        self.__saver__ = tf.train.Saver(max_to_keep=3)
        self.__writer__ = tf.summary.FileWriter(logs_dir, graph=tf.get_default_graph())

    @property
    def saver(self):
        return self.__saver__

    @property
    def writer(self):
        return self.__writer__





