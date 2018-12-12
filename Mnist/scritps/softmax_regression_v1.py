# coding=utf-8
# date: 2018/12/11, 14:32
# name: smz

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

def read_mnist(data_dir):
    return input_data.read_data_sets(data_dir)


def show_mnist(Mnist):
    """<Mnist>show"""
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    img = Mnist.train.images[0]             # images_shape: (55000, 784)
    img = img.reshape(-1, 28)
    ax.axis("off")
    ax.imshow(img)

    plt.show()


if __name__ == "__main__":
    data_dir = "J:\TF-deep-learn\Mnist\data\Mnist_data"
    mnist = read_mnist(data_dir)
    show_mnist(mnist)