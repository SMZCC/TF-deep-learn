# coding=utf-8
# date: 2018/12/19, 13:49
# name: smz


import tensorflow as tf


def demo_one():
    """defference between softmax and log_softmax"""
    array_1 = tf.convert_to_tensor([1, 2, 3, 4, 5], dtype=tf.float32)

    single_softmax = tf.nn.softmax(array_1, name='single_softmax')
    log_softmax = tf.nn.log_softmax(array_1, name="log_softmax")
    log_array = tf.log(array_1, name='log_values')
    log_single_softmax = tf.log(single_softmax, name="log_single_softmax")

    with tf.Session() as sess:
        single_softmax_value, log_softmax_value, log_array_values, log_single_softmax_value = sess.run(
            fetches=[single_softmax, log_softmax, log_array, log_single_softmax])


        print("single_softmax_value:{}\n".format(single_softmax_value))
        print("log_softmax_value:{}\n".format(log_softmax_value))
        print("log_array_values:{}\n".format(log_array_values))
        print("log_single_softmax:{}\n".format(log_single_softmax_value))


if __name__ == "__main__":
    demo_one()