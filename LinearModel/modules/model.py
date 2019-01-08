# coding=utf-8
# date: 2018/12/24, 14:53
# name: smz

import tensorflow as tf


class TumorModel(object):
    def __init__(self, opts):
        self.opts = opts
        self.inputs = tf.placeholder(dtype=tf.float32, shape=(None, 2), name="Input")
        self.labels = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="Labels")
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.__saver__ = None
        self.__writer__ = None

    def build(self):
        with tf.variable_scope("params") as params:
            weights = tf.get_variable(name="weights", dtype=tf.float32, initializer=tf.truncated_normal_initializer(), shape=(2, 1))
            biases = tf.get_variable(name="biases", dtype=tf.float32, initializer=tf.constant_initializer(0.0), shape=(1, ))

        logits = tf.matmul(self.inputs, weights) + biases
        self.probability = tf.nn.sigmoid(logits)
        self.loss = tf.reduce_mean(- self.labels * tf.log(self.probability) - (1 - self.labels) * tf.log(1 - self.probability))
        tf.summary.scalar('Loss', self.loss)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.opts["learning_rate"])
        self.train_step = self.optimizer.minimize(self.loss, global_step=self.global_step)
        self.__saver__ = tf.train.Saver(max_to_keep=3)
        self.__writer__ = tf.summary.FileWriter(logdir=self.opts["log_dir"], graph=tf.get_default_graph())
        self.merge_op = tf.summary.merge_all()

    @property
    def saver(self):
        return self.__saver__

    @property
    def writer(self):
        return self.__writer__


