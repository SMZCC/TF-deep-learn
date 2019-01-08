# coding=utf-8
# date: 2019/1/7, 18:09
# name: smz

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap, colorConverter

from LinearModel.configuration.options import opts
from LinearModel.modules.model3 import ModelThreeClasses


def show_3_classes():
    train_x_200 = np.load("../data/train_data_X3.npy")[:200, :]
    train_y_200 = np.load("../data/train_data_Y3.npy")[:200, :]
    colors = ['r' if np.argmax(label) == 0 else 'b' if np.argmax(label) ==1 else 'y' for label in train_y_200]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.scatter(train_x_200[:, 0], train_x_200[:, 1], c=colors)

    xs = np.linspace(-2, 8, 200)
    ys = np.linspace(-2, 8, 200)
    xs, ys = np.meshgrid(xs, ys)

    class_plane = np.zeros((200, 200))


    model3cs = ModelThreeClasses(opts)
    model3cs.build()
    model_name = "model3s.ckpt"

    with tf.Session() as sess:
        model3cs.saver.restore(sess, opts["checkpoints_dir"] +model_name + "-8000")
        for i in range(200):
            for j in range(200):
                feed_dict = {model3cs.inputs: [[xs[i, j], ys[i, j]]]}
                softmax_vlue = sess.run(fetches=[model3cs.softmax_outs], feed_dict=feed_dict)
                c = np.argmax(softmax_vlue[0], axis=1)
                class_plane[i, j] = c[0]


    cmap = ListedColormap([
        colorConverter.to_rgba('r', alpha=0.30),
        colorConverter.to_rgba('b', alpha=0.30),
        colorConverter.to_rgba('y', alpha=0.30),
        colorConverter.to_rgba('c', alpha=0.30)  # 高度被平均分为4份，每份对应一种颜色
    ]
    )

    ax.contourf(xs, ys, class_plane, cmap=cmap)
    plt.show()


if __name__ == "__main__":
    show_3_classes()