# coding=utf-8
# date: 2018/12/11, 14:04
# name: smz

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

if __name__ == "__main__":
    mnist = input_data.read_data_sets("Mnist_data", one_hot=True)

