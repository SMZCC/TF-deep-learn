# coding=utf-8
# date: 2018/11/26, 14:44
# name: smz

import tensorflow as tf


def demo_one():
    indices = [0, 1, 2, -1]   # -1轴表示全部为0,即没有一个是标签
    depth = 4
    on_value = 1
    off_value = 0
    tensor = tf.one_hot(indices=indices, depth=depth, on_value=on_value, off_value=off_value, axis=1)

    with tf.Session() as sess:
        tensor_value = sess.run(tensor)
        print("one_hot tensor:\n", tensor_value)



if __name__ == "__main__":
    demo_one()
