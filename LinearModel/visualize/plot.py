# coding=utf-8
# date: 2018/12/24, 16:22
# name: smz

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from LinearModel.modules.model import TumorModel
from LinearModel.configuration.options import opts
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


def check_results():
    check_path = "J:\\TF-deep-learn\\LinearModel\\checkpoints\\tumor_model-600"
    print_tensors_in_checkpoint_file(file_name=check_path, tensor_name=None, all_tensors=True)
    """params/biases, params/weights"""


def plot_results():
    """原表达式为：z = w1x1 + w2x2+b，现将其投影到由x1,与x2构成的平面上来,有：x2 = -x1*w1/w2 - b/w2"""
    data_X = np.load("../data/train_data_X.npy")
    data_Y = np.load("../data/train_data_Y.npy")
    tumor_model = TumorModel(opts)
    tumor_model.build()
    with tf.Session() as sess:
        tumor_model.saver.restore(sess, opts["checkpoints_dir"]+"tumor_model-600")
        g = tf.get_default_graph()
        weights = g.get_tensor_by_name("params/weights:0")
        biases = g.get_tensor_by_name("params/biases:0")
        weights_value, biases_value = sess.run(fetches=[weights, biases])
        print("weights:{}\n".format(weights_value))
        print("biases:{}\n".format(biases_value))

        colors = ['r' if label == 0 else 'b' for label in data_Y]

        x = np.linspace(-2, 8, 200)   # 用于绘制直线的变量x
        y = -  x * weights_value[0] / weights_value[1] - biases_value / weights_value[1]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(data_X[:, 0], data_X[:, 1], c=colors)
    ax.plot(x, y)
    plt.show()


if __name__ == "__main__":
    plot_results()