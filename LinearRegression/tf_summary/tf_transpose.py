# coding=utf-8
# date: 2018/12/10, 10:14
# name: smz

import tensorflow as tf

def demo_transpose():
    matrix_a = tf.constant([[[1, 2], [3, 4], [5, 6]]], dtype=tf.float32)
    matrix_b = tf.transpose(matrix_a)
    matrix_a_shape = tf.shape(matrix_a)
    matrix_b_shape = tf.shape(matrix_b)
    with tf.Session() as sess:
        print("matrix_b:\n", sess.run(matrix_b))
        print("matrix_a_shape:", sess.run(matrix_a_shape))
        print("matrix_b_shape:", sess.run(matrix_b_shape))

if __name__ == "__main__":
    demo_transpose()