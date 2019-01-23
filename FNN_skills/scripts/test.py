# coding=utf-8
# date: 2019/1/21, 16:12
# name: smz

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from FNN_skills.modules.model import ModelV1
from FNN_skills.configration.options import opts


def test(model=None):
    test_0_x = np.load("../data/class_0_test_X.npy")
    test_1_x = np.load("../data/class_1_test_X.npy")
    test_2_x = np.load("../data/class_2_test_X.npy")
    test_3_x = np.load("../data/class_3_test_X.npy")
    test_X = np.concatenate([test_0_x, test_1_x, test_2_x, test_3_x], axis=0)

    test_0_y = np.load("../data/class_0_test_Y.npy")
    test_1_y = np.load("../data/class_1_test_Y.npy")
    test_2_y = np.load("../data/class_2_test_Y.npy")
    test_3_y = np.load("../data/class_3_test_Y.npy")
    test_Y = np.concatenate([test_0_y, test_1_y, test_2_y, test_3_y], axis=0)

    test_x, test_y = shuffle(test_X, test_Y)

    num_tests = len(test_y)

    model_v1 = ModelV1(opts)
    model_v1.build(False)

    with tf.Session() as sess:
        if model is None:
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            print("restoring model from {}".format(model))
            model_v1.saver.restore(sess, model)

        fetches = [model_v1.loss, model_v1.preds]
        feed_dict = {
            model_v1.inputs: test_x,
            model_v1.labels: test_y
        }

        loss_value, pred_value = sess.run(fetches, feed_dict)
        pred_idxs = np.argmax(pred_value, axis=1)
        labels_idxs = np.argmax(test_y, axis=1)
        accuracy = float(np.sum(np.asarray(pred_idxs == labels_idxs))) / num_tests
        print("test loss:%.6f, accuracy:%.6f"%(loss_value, accuracy))


if __name__ == "__main__":
    test("../checkpoints/plan_e/plan_e.ckpt-12000")