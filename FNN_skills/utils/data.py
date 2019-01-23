# coding=utf-8
# date: 2019/1/12, 13:59
# name: smz

import os
import numpy as np
from PIL import Image
from sklearn.utils import shuffle


def load_mnist_data(datasets_dir, data_paths_file):
    """
    读取本地磁盘上png格式的mnist数据
    :param datasets_dir: string
    :param data_paths_file:  string
    :return:  data:<List>, labels:<List>
    """
    data = []
    labels = []
    with open(data_paths_file, 'r') as f:
        for line in f:
            img_path_, label = line.split(' ')
            img_path = os.path.join(datasets_dir, img_path_)
            image = Image.open(img_path).convert("RGB")
            image = np.asarray(image)
            data.append(image)
            labels.append(label)

    return data, labels


def gen_data(num_classes, num_samples, means, covs, diffs, one_hot=True):
    """
    用于生成num_classes种类
    :param num_classes:  类别数目      <Int>
    :param num_samples:  样本的总数    <Int>
    :param diffs:   各个样本之间的偏移, <List>
    :param means:   样本维度的均值,     <ndarray/list>
    :param covs:    样本维度间的协方差矩阵 <ndarray>
    :param one_hot: 是否使用one_hot标识   bool
    :return:  X, Y  <ndarray>
    """
    size_per_class = num_samples // num_classes
    x = np.random.multivariate_normal(means, covs, size_per_class)
    y = np.zeros((size_per_class, 1))

    for idx, diff in enumerate(diffs):
        x_ = np.random.multivariate_normal(means+diff, covs, size_per_class)
        y_ = (idx + 1) * np.ones((size_per_class, 1))

        x = np.concatenate([x, x_], axis=0)
        y = np.concatenate([y, y_], axis=0)


    if one_hot:
        labels = np.asarray([y == label for label in range(num_classes)], dtype=np.float32)
        labels = np.hstack(labels)
        X, Y = shuffle(x, labels)
    else:
        X, Y = shuffle(x, y)

    return X, Y


def gen_one_class_data(num_samples, means, covs, label, classes, one_hot=True):
    X = np.random.multivariate_normal(mean=means, cov=covs, size=num_samples)
    if one_hot:
        Y = np.zeros((num_samples, classes))
        Y[:, label] = 1
    else:
        Y = label * np.ones((num_samples, 1))
    return X, Y




if __name__ == "__main__":
    # datasets_dir = "J:/data_sets/"
    # data_paths_file = "../data/train.txt"
    # load_mnist_data(datasets_dir, data_paths_file)

    num_classes = 2
    num_samples = 100
    num_fields = 2
    means = np.random.rand(num_fields)
    covs = np.eye(num_fields, num_fields)
    diffs = [[0.3, 0.3]]
    X, Y = gen_data(num_classes, num_samples, means, covs, diffs, False)
