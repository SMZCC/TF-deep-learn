# coding=utf-8
# date: 2018/11/22, 15:15
# name: smz

import tensorflow as tf
import numpy as np
from collections import OrderedDict


config = OrderedDict()
config['save_path'] = "J:\\TF-deep-learn\\LinearRegression\\saved\\regress_v2\\regressor_v2"
config['log_dir'] = "J:\\TF-deep-learn\\LinearRegression\\logs\\regress_v2\\"
config['batch_size'] = 100
config['epochs'] = 20000


def gen_data():
    # 第一部分：生成数据
    x_train = np.linspace(-1, 1, 100)
    y_train = x_train * 2 + np.random.randn(*x_train.shape) * 0.3  # 产生样本的时候，让每个样本产生一些偏差
    return x_train, y_train


class LinearRegressor(object):
    def __init__(self, config=config):
        self.pre = None
        self.loss = None
        self.optimizer = None
        self.saver = None
        self.file_witer = None
        self.graph = None
        self.train_step = None
        self.merge_op = None
        self.__complete__ = False

        self.global_step = tf.Variable(0, trainable=False)
        self.inputs = OrderedDict()
        self.config = config

    @property
    def complete(self):
        return self.__complete__

    @complete.setter
    def complete(self, value):
        """value： bool"""
        self.__complete__ = value

    def __add_saver__(self):
        assert self.complete, "The saver must be added after the graph building---SMZ"
        self.saver = tf.train.Saver(max_to_keep=3)

    def __add_file_writer__(self):
        assert self.graph is not None, 'The graph must be gotten ---SMZ'
        self.file_writer = tf.summary.FileWriter(logdir=self.config['log_dir'], graph=self.graph)

    def build(self):
        self.inputs['train_x'] = tf.placeholder(dtype=tf.float32, shape=[self.config['batch_size'], ], name='train_x')
        self.inputs['train_label'] = tf.placeholder(dtype=tf.float32, shape=[self.config['batch_size'], ], name='train_label')

        weight = tf.Variable([1], dtype=tf.float32,  name='weight')
        bias = tf.Variable([0.1], dtype=tf.float32,  name='bias')

        self.pre = tf.multiply(weight, self.inputs['train_x']) + bias
        self.loss = tf.reduce_mean(tf.square(self.pre-self.inputs['train_label']))
        tf.summary.scalar('loss', self.loss)

        self.complete = True
        self.__add_saver__()
        self.graph = tf.get_default_graph()
        self.__add_file_writer__()

        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.train_step = self.optimizer.minimize(self.loss, global_step=self.global_step)
        self.merge_op = tf.summary.merge_all()

    def train(self, train_xs, labels):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            feed_dict = {self.inputs['train_x']:train_xs, self.inputs['train_label']:labels}
            for i in range(self.config['epochs']):
                loss_value, _, merge_str, global_step_value = sess.run([self.loss, self.train_step, self.merge_op, self.global_step],
                                                                       feed_dict=feed_dict)
                print("Epoch:%d, loss:%.6f\n"%(i, loss_value))
                self.file_writer.add_summary(merge_str, global_step=global_step_value)
                if (i+1) % 100 == 0:
                    self.saver.save(sess, self.config['save_path'] + '.ckpt', global_step=self.global_step)
        self.file_writer.close()


if __name__ == "__main__":
    train_x, labels = gen_data()
    model = LinearRegressor(config)
    model.build()
    model.train(train_x, labels)









