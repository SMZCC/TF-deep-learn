# coding=utf-8
# date: 2019/1/14, 20:52
# name: smz


import numpy as np
import matplotlib.pyplot as plt


def plot_train_data():
    data_x = np.load("../data/train_X.npy")
    data_y = np.load("../data/train_Y.npy")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    xrs = np.empty((0, 2))
    xbs = np.empty((0, 2))

    for (x, y) in zip(data_x, data_y):
        if y == 0:
            x = np.expand_dims(x, axis=0)
            xrs = np.concatenate([xrs, x], axis=0)
        else:
            x = np.expand_dims(x, axis=0)
            xbs = np.concatenate([xbs, x], axis=0)

    xrs = np.asarray(xrs, dtype=np.float32)
    xbs = np.asarray(xbs, dtype=np.float32)
    ax.scatter(xrs[:, 0], xrs[:, 1], c='r', marker='+')
    ax.scatter(xbs[:, 0], xbs[:, 1], c='b', marker='o')
    plt.show()


def plot_four_classes():
    class_0_x = np.load("../data/class_0_train_X.npy")
    class_0_y = np.load("../data/class_0_train_Y.npy")
    class_1_x = np.load("../data/class_1_train_X.npy")
    class_1_Y = np.load("../data/class_1_train_Y.npy")
    class_2_x = np.load("../data/class_2_train_X.npy")
    class_2_Y = np.load("../data/class_2_train_Y.npy")
    class_3_x = np.load("../data/class_3_train_X.npy")
    class_3_y = np.load("../data/class_3_train_y.npy")

    class_0_test_x = np.load("../data/class_0_test_X.npy")
    class_0_test_y = np.load("../data/class_0_test_Y.npy")
    class_1_test_x = np.load("../data/class_1_test_X.npy")
    class_1_test_y = np.load("../data/class_1_test_y.npy")
    class_2_test_x = np.load("../data/class_2_test_X.npy")
    class_2_test_y = np.load("../data/class_2_test_Y.npy")
    class_3_test_x = np.load("../data/class_3_test_X.npy")
    class_3_test_y = np.load("../data/class_3_test_Y.npy")

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("train_data")
    ax.scatter(class_0_x[:, 0], class_0_x[:, 1], c='r', marker='.')
    ax.scatter(class_1_x[:, 0], class_1_x[:, 1], c='b', marker='+')
    ax.scatter(class_2_x[:, 0], class_2_x[:, 1], c="y", marker='x')
    ax.scatter(class_3_x[:, 0], class_3_x[:, 1], c='c', marker='p')

    fig2 = plt.figure()
    ax = fig2.add_subplot(1, 1, 1)
    ax.set_title("test_data")
    ax.scatter(class_0_test_x[:, 0], class_0_test_x[:, 1], c='r', marker='.')
    ax.scatter(class_1_test_x[:, 0], class_1_test_x[:, 1], c='b', marker='+')
    ax.scatter(class_2_test_x[:, 0], class_2_test_x[:, 1], c='y', marker='x')
    ax.scatter(class_3_test_x[:, 0], class_3_test_x[:, 1], c='c', marker='p')

    plt.show()


if __name__ == "__main__":
    # plot_train_data()
    plot_four_classes()