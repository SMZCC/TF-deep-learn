# coding=utf-8
# date: 2018/12/19, 16:46
# name: smz


import tensorflow as tf


def demo_one():
    logits = tf.constant([[12., 8., 9.], [2., 4., 8.]], dtype=tf.float32, name="logits")
    labels = tf.constant([[0., 1., 0.], [0., 0., 1.]], dtype=tf.float32, name="labels")

    softmax = tf.nn.softmax(logits, name="softmax")
    softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name="softmax_cross_entropy")  # tf自带的交叉熵api

    self_define_cross_entropy = - tf.reduce_sum(labels * tf.log(softmax), axis=1)   # 只有交叉熵公式的前半部分
    self_complete_cross_entropy = - tf.reduce_sum(labels * tf.log(softmax) + (1. - labels) * tf.log(1 - softmax), axis=1)  # 按照交叉熵的公式写

    with tf.Session() as sess:
        softmax_value, softmax_corss_entropy_value, self_define_cross_entropy_value, self_complete_cross_entropy_value = sess.run(
            fetches=[softmax, softmax_cross_entropy, self_define_cross_entropy, self_complete_cross_entropy]
        )
        print("softmax_value:{}\n".format(softmax_value))
        print("softmax_cross_entropy_value:{}\n".format(softmax_corss_entropy_value))
        print("self_define_cross_entropy_value:{}\n".format(self_define_cross_entropy_value))
        print("self_complete_cross_entropy_value:{}\n".format(self_complete_cross_entropy_value))

def demo_two():
    logits = tf.constant([1, 2, 3, 4], dtype=tf.float32, name="logits")
    labels = tf.constant([0, 1, 1, 1], dtype=tf.float32, name="labels")

    sigmoid_logits = tf.nn.sigmoid(logits, name="sigmoid_logits")

    sigmoid_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels, name="sigmoid_cross_entropy")
    self_sigmoid_cross_entropy = - labels * tf.log(sigmoid_logits) - (1 - labels) * tf.log(1 - sigmoid_logits)

    with tf.Session() as sess:
        sigmoid_cross_entropy_value, self_sigmoid_cross_entropy_value = sess.run(
            fetches=[sigmoid_cross_entropy, self_sigmoid_cross_entropy]
        )
    print("sigmoid_cross_entropy_value:{}\n".format(sigmoid_cross_entropy_value))
    print("self_sigmoid_cross_entropy_value:{}\n".format(self_sigmoid_cross_entropy_value))

tf.random_uniform_initializer()
if __name__ == "__main__":
    demo_two()