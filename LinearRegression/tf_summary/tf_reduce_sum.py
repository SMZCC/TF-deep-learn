# coding=utf-8
# date: 2018/12/12, 10:00
# name: smz

import tensorflow as tf


def demo_one():
    array = tf.convert_to_tensor([[1, 2], [3, 4]])
    sum_result = tf.reduce_sum(array, reduction_indices=1)   # 相当于axis=1
    with tf.Session() as sess:
        sum = sess.run(sum_result)

    print("array_sum:", sum)


if __name__ == "__main__":
    demo_one()