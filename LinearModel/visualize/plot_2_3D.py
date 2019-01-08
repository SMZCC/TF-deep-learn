# coding=utf-8
# date: 2019/1/8, 16:02
# name: smz

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap, colorConverter

from LinearModel.modules.model import TumorModel
from LinearModel.configuration.options import opts


def show_3D_2_classes():
    data_x = np.load("../data/train_data_X.npy")[:200, :]
    data_y = np.load("../data/train_data_Y.npy")[:200]
    colors = ['r' if label == 0 else 'b'  for label in data_y]

    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1, projection='3d')

    ax.scatter(data_x[:, 0], data_x[:, 1], 0, c=colors)

    x = np.linspace(-2, 8, 100)
    y = np.linspace(-2, 8, 100)
    xs, ys = np.meshgrid(x, y)

    model_name = "tumor_model-600"
    model_path = opts["checkpoints_dir"] + model_name

    class_plane = np.zeros((100, 100))

    with tf.Session() as sess:
        tumor_model = TumorModel(opts)
        tumor_model.build()
        tumor_model.saver.restore(sess, model_path)

        for i in range(100):
            for j in range(100):
                feed_dict = {
                    tumor_model.inputs:[[xs[i, j], ys[i, j]]]
                }
                fetches = [tumor_model.probability]

                probability_ = sess.run(fetches=fetches, feed_dict=feed_dict)
                probability = np.squeeze(probability_)
                class_plane[i, j] = probability


    cmap = ListedColormap([
        colorConverter.to_rgba('r', alpha=0.3),
        colorConverter.to_rgba('b', alpha=0.3)
        # colorConverter.to_rgba('y', alpha=0.3),
        # colorConverter.to_rgba('c', alpha=0.3)
    ])
    ax.contourf(xs, ys, class_plane, cmap=cmap)
    plt.show()


if __name__ == "__main__":
    show_3D_2_classes()





