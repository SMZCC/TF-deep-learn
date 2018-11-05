# coding=utf-8
# date: 2018/11/5, 18:40
# name: smz

import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


def demo_one():
    W = tf.Variable(4, name='W_op')
    # saver = tf.train.Saver({v.op.name: v for v in [W]})   # tensor_name:  W_op，这里可以看出我们命名的对象其实是操作
    #                                                       # 4

    saver = tf.train.Saver({"W": W})   # tensor_name:  W ，要保存的变量的名字可以由自己随意命名
                                       # 4
    save_path = "J:\\TF-deep-learn\\LinearRegression\\saved\\saver_exercise\\saver_exercise_1.ckpt"

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver.save(sess, save_path)


if __name__ == "__main__":
    demo_one()
    save_path = "J:\\TF-deep-learn\\LinearRegression\\saved\\saver_exercise\\saver_exercise_1.ckpt"
    print_tensors_in_checkpoint_file(save_path, None, True)