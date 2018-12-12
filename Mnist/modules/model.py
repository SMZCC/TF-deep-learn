# coding=utf-8
# date: 2018/12/11, 16:34
# name: smz


import tensorflow as tf


class MnistSoftmax(object):
    def __init__(self):
        self.__complete__ = False
        self.__writer__ = None
        self.__saver__ = None

        self.input = tf.placeholder(dtype=tf.float32, shape=[None, 784], name="Input")
        self.label = tf.placeholder(dtype=tf.float32, shape=[None, 10], name="Label")

    def build(self, logs_dir, checkpoints_dir):
        with tf.variable_scope("params") as param:
            self.weights = tf.get_variable(name="weights", shape=[784, 10], initializer=tf.random_normal_initializer())
            self.biases = tf.get_variable(name="biases", shape=[10], initializer=tf.constant_initializer(0.1))

        self.logits = tf.matmul(self.input, self.weights) + self.biases
        self.pred = tf.nn.softmax(self.logits)   # 使用softmax来预测类别
        self.loss =
