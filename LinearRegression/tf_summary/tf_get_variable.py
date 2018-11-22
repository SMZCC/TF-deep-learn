# coding=utf-8
# date: 2018/11/6, 18:15
# name: smz

import tensorflow as tf


def demo_one():
    var_1 = tf.get_variable('var_1', shape=[1], dtype=tf.float32, initializer=tf.constant_initializer)
    var_2 = tf.get_variable('var_1', shape=[1], dtype=tf.float32, initializer=tf.constant_initializer)  # 报错，tf.get_variable不能定义同名变量

    print('var_1:', var_1)
    print('var_2:', var_2)


if __name__ == "__main__":
    demo_one()