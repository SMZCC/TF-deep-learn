# coding=utf-8
# date: 2018/12/12, 10:54
# name: smz

import tensorflow as tf

def demo_one():
    tensor = tf.convert_to_tensor([[1,2], [3, 4], [5, 6]])
    _shuffled_tensor = tf.random_shuffle(tensor)
    with tf.Session() as sess:
        shuffled_tensor = sess.run(_shuffled_tensor)
        print("shuffled_tensor:", shuffled_tensor)  # 0轴内元素随机打乱


if __name__ == "__main__":
    demo_one()