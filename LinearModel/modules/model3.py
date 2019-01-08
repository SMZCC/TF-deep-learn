# coding=utf-8
# date: 2019/1/5, 14:53
# name: smz

import tensorflow as tf


class ModelThreeClasses(object):
    def __init__(self, opts):
        """opts: 配置类"""
        self.inputs = tf.placeholder(dtype=tf.float32, name="Inputs", shape=(None, 2))
        self.labels = tf.placeholder(dtype=tf.float32, name="Labels", shape=(None, 3))
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.opts = opts
        self.__saver = None
        self.__writer = None

    def build(self):
        weights = tf.get_variable(name="weights", shape=(2, 3), initializer=tf.truncated_normal_initializer(), dtype=tf.float32)
        biases = tf.get_variable(name="biases", shape=(3, ), initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        self.logits = tf.matmul(self.inputs, weights) + biases
        self.softmax_outs = tf.nn.softmax(self.logits)

        # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels, name="SoftmaxLoss"), axis=0)
        self.my_softmax_cross_entropy = tf.reduce_sum(- self.labels * tf.log(self.softmax_outs), axis=1, name="softmax_cross_entropy")

        self.loss = tf.reduce_mean(self.my_softmax_cross_entropy, name="loss_last")
        tf.summary.scalar(name="loss", tensor=self.loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.opts["learning_rate"])
        self.train_step = optimizer.minimize(self.loss, global_step=self.global_step)

        self.merge_op = tf.summary.merge_all()
        self.__saver = tf.train.Saver(var_list=None, max_to_keep=3)
        self.__writer = tf.summary.FileWriter(logdir=self.opts["log_dir"], graph=tf.get_default_graph())

    @property
    def saver(self):
        return self.__saver

    @property
    def writer(self):
        return self.__writer