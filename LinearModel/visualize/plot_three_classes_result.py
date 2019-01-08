# coding=utf-8
# date: 2019/1/5, 19:11
# name: smz

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from LinearModel.configuration.options import opts
from LinearModel.modules.model3 import ModelThreeClasses
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


def check_saves():
    model_name = "model3s.ckpt"
    file_path = opts["checkpoints_dir"] + model_name + "-8000"
    print_tensors_in_checkpoint_file(file_name=file_path, tensor_name=None, all_tensors=True)
    """weights, biases"""


def plot_3_classes_results():
    data_x = np.load("../data/train_data_X3.npy")
    data_y = np.load("../data/train_data_Y3.npy")  # [[0, 0, 1], [1, 0, 0], ...]

    colors = ['r' if np.argmax(label) == 0 else 'b' if np.argmax(label) == 1 else 'y' for label in data_y]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(data_x[:, 0], data_x[:, 1], c=colors)

    model3 = ModelThreeClasses(opts)
    model3.build()
    model_name = "model3s.ckpt"
    file_path = opts["checkpoints_dir"] + model_name + "-8000"

    with tf.Session() as sess:
        model3.saver.restore(sess, file_path)

        g = tf.get_default_graph()
        weights = g.get_tensor_by_name("weights:0")
        biases = g.get_tensor_by_name("biases:0")
        weights_value, biases_value = sess.run(fetches=[weights, biases])

    x = np.linspace(-2, 8, 200)

    y = - x * weights_value[0][0] / weights_value[1][0]  - biases_value[0] / weights_value[1][0]
    ax.plot(x, y, 'b-', label="line1", lw=1)

    y = - x * weights_value[0][1] / weights_value[1][1] - biases_value[1] / weights_value[1][1]
    ax.plot(x, y, 'g-', label="line2", lw=2)

    y = - x * weights_value[0][2] / weights_value[1][2] - biases_value[2] /weights_value[1][2]
    ax.plot(x, y, 'r-', label="line3", lw=3)

    plt.legend(loc=1)
    plt.show()


if __name__ == "__main__":
    # check_saves()
    plot_3_classes_results()