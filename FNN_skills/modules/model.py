# coding=utf-8
# date: 2019/1/12, 13:59
# name: smz

import tensorflow as tf


class ModelV1(object):
    def __init__(self, opts):
        self.inputs = tf.placeholder(dtype=tf.float32, shape=(None, opts["input_fields"]), name="Inputs")
        self.labels = tf.placeholder(dtype=tf.float32, shape=(None, opts["label_fields"]), name="Labels")
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.opts = opts
        self.loss = None
        self.train_step = None
        self.__saver = None
        self.__writer = None

    def build(self, isTrain=True):
        """使用字典构建参数"""
        with tf.variable_scope("params") as params:
            weights = {
                "h1":tf.get_variable(name="h1_weights", dtype=tf.float32,
                                     shape=(self.opts["input_fields"], self.opts["hidden_fields"][0]),
                                     initializer=tf.truncated_normal_initializer(stddev=0.1)),
                "h2":tf.get_variable(name="h2_weights", dtype=tf.float32,
                                     shape=(self.opts["hidden_fields"][0], self.opts["label_fields"]),
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
            }

            biases = {
                "h1": tf.get_variable(name="h1_biases", dtype=tf.float32,
                                      shape=(self.opts["hidden_fields"][0], ),
                                      initializer=tf.constant_initializer(0.)),
                "h2": tf.get_variable(name="h2_biases", dtype=tf.float32,
                                      shape=(self.opts["label_fields"], ),
                                      initializer=tf.constant_initializer(0.))
            }

        layer1 = tf.nn.relu(tf.add(tf.matmul(self.inputs, weights["h1"]), biases["h1"]), name="Layer1_Relu")
        layer2 = tf.nn.tanh(tf.add(tf.matmul(layer1, weights["h2"]), biases["h2"]), name="Layer2_tanh")
        self.preds = tf.nn.softmax(layer2, name="softmax")
        # layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, weights["h2"]), biases["h2"]), name="Layer2_relu")

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layer2, labels=self.labels), name="SCE_Loss")
        tf.summary.scalar("loss", self.loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.opts["learning_rate"])
        self.train_step = optimizer.minimize(self.loss, global_step=self.global_step)
        self.merge_op = tf.summary.merge_all()

        self.__saver = tf.train.Saver(var_list=None, max_to_keep=self.opts["max_to_keep"])
        if isTrain:
            self.__writer = tf.summary.FileWriter(logdir=self.opts["logs_dir"], graph=tf.get_default_graph())

    @property
    def saver(self):
        return self.__saver

    @property
    def writer(self):
        return self.__writer


