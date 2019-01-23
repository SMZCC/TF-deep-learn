# coding=utf-8
# date: 2019/1/14, 16:08
# name: smz

import numpy as np
from FNN_skills.utils.data import gen_data
from FNN_skills.utils.data import gen_one_class_data


def generate_data():
    num_classes = 4
    num_fields = 2
    num_samples = 320
    means = np.random.rand(num_fields)
    covs = np.eye(num_fields, num_fields)
    diffs = [[3.0, 0.], [3.0, 3.0], [0., 3.0]]
    X, Y = gen_data(num_classes, num_samples, means, covs, diffs, False)
    Y = Y % 2
    np.save("../data/train_X.npy", X)
    np.save("../data/train_Y.npy", Y)


def generate_four_classes():
    num_fields = 2
    means = np.random.rand(num_fields)
    covs = np.eye(num_fields, num_fields)
    num_samples_train = 10000
    num_samples_test = 2500

    classes = 4

    for class_ in range(classes):
        if class_ == 0:
            X, Y = gen_one_class_data(num_samples_train, means, covs, class_, classes)
            test_X, test_Y = gen_one_class_data(num_samples_test, means, covs, class_, classes)
        elif class_ == 1:
            X, Y = gen_one_class_data(num_samples_train, means + [10., 0.], covs, class_, classes)
            test_X, test_Y = gen_one_class_data(num_samples_test, means + [10., 0.], covs, class_, classes)
        elif class_ == 2:
            X, Y = gen_one_class_data(num_samples_train, means + [10., 10.], covs, class_, classes)
            test_X, test_Y = gen_one_class_data(num_samples_test, means + [10., 10.], covs, class_, classes)
        elif class_ == 3:
            X, Y = gen_one_class_data(num_samples_train, means + [0., 10.], covs, class_, classes)
            test_X, test_Y = gen_one_class_data(num_samples_test, means + [0., 10.], covs, class_, classes)

        np.save("../data/means.npy", np.asarray(means))
        np.save("../data/class_%s_train_X.npy"%(class_), X)
        np.save("../data/class_%s_train_Y.npy"%(class_), Y)
        np.save("../data/class_%s_test_X.npy"%(class_), test_X)
        np.save("../data/class_%s_test_Y.npy"%(class_), test_Y)


if __name__ == "__main__":
    # generate_data()
    generate_four_classes()





