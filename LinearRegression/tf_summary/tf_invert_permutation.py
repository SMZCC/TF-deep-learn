# coding=utf-8
# date: 2018/12/10, 10:59
# name: smz

import tensorflow as tf

def demo_invert_permutation():
    array1 = tf.constant([0, 1, 2, 3, 4], dtype=tf.int32)
    array2 = tf.invert_permutation(array1)
    with tf.Session() as sess:
        print(sess.run(array2))


if __name__ == "__main__":
    demo_invert_permutation()