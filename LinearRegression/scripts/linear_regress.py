# coding=utf-8
# date: 2018/11/1, 17:08
# name: smz

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
"""真实的模型为：y=2x"""


def gen_data():
    # 第一部分：生成数据
    x_train = np.linspace(-1, 1, 100)
    y_train = x_train * 2 + np.random.randn(*x_train.shape) * 0.3  # 产生样本的时候，让每个样本产生一些偏差
    return x_train, y_train


def main():
    x_train, y_train = gen_data()
    # 第二部分：模型搭建
    x = tf.placeholder(shape=(100, ), dtype=tf.float32)    # 100 个样本，每个样本是一个scalar
    y = tf.placeholder(shape=(100, ), dtype=tf.float32)

    num_step = tf.Variable(0, trainable=False, name='num_step')
    _W = tf.Variable(tf.random_normal([1, ]), name='W')   # 对于一个样本来说，权重就应该是一个scalar，所以W的shape=(1,)
    _b = tf.Variable([0.1], name='b')
    out_ = tf.multiply(x, _W) + _b  # 与matmul的区别在于这个每个数都是乘以一样的W
    loss_ = tf.reduce_mean(tf.square(out_ - y))

    lr = 0.001
    optimizer = tf.train.GradientDescentOptimizer(lr)
    train_step = optimizer.minimize(loss_, global_step=num_step)

    # 训练模型
    EPOCHS = 20000
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()   # 必须要在图定义完毕之后生成该对象，否则没有可以保存的变量
    save_path = 'J:\\TF-deep-learn\\LinearRegression\\saved\\linear_model.ckpt'
    with tf.Session() as sess:
        sess.run(fetches=[init])
        for epoch in range(EPOCHS):
            loss, _, W, b = sess.run([loss_, train_step, _W, _b], feed_dict={x: x_train, y: y_train})
            print("Epoch:{}, loss:{:.6f}, W:{}, B:{}".format(epoch, loss, W[0], b[0]))

        saver.save(sess, save_path=save_path, global_step=num_step)


def show_data(x, y, label='original data'):
    fig = plt.figure()
    ax = plt.Axes(fig, (0., 0., 1., 1.))
    fig.add_axes(ax)
    ax.plot(x, y, 'ro', label=label)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
