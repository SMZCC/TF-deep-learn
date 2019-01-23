# coding=utf-8
# date: 2019/1/21, 14:18
# name: smz

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

from FNN_skills.modules.model import ModelV1
from FNN_skills.configration.options import opts


def split_data(data, ratios, batch_size):
    ratio1, ratio2, ratio3, ratio4 = ratios.split(":")
    num_data = len(data)
    sum_ = float(ratio1) + float(ratio2) + float(ratio3) + float(ratio4)
    num_ratio1 = int(batch_size * float(ratio1) / sum_)
    num_ratio2 = int(batch_size * float(ratio2) / sum_)
    num_ratio3 = int(batch_size * float(ratio3) / sum_)
    num_ratio4 = int(batch_size * float(ratio4) / sum_)

    pos = 0
    num_ratios = [num_ratio1, num_ratio2, num_ratio3, num_ratio4]
    batch_data = []
    while pos < num_data:
        try:
            for num_ratio in num_ratios:
                batch_data.append(data[pos: pos+num_ratio])
                pos = pos + num_ratio
        except:
            pass

    return batch_data


def train():
    """plan_e: batch_size=100,4个bath一个循环
    1:2:3:4, 一份是10
    2:3:4:1
    3:4:1:2
    4:1:2:3
    """
    train_0_x = np.load("../data/class_0_train_X.npy")
    train_0_y = np.load("../data/class_0_train_Y.npy")
    train_1_x = np.load("../data/class_1_train_X.npy")
    train_1_y = np.load("../data/class_1_train_Y.npy")
    train_2_x = np.load("../data/class_2_train_X.npy")
    train_2_y = np.load("../data/class_2_train_Y.npy")
    train_3_x = np.load("../data/class_3_train_X.npy")
    train_3_y = np.load("../data/class_3_train_y.npy")

    batch_0_xs = split_data(train_0_x, "1:2:3:4", opts["batch_size"])
    batch_0_ys = split_data(train_0_y, "1:2:3:4", opts["batch_size"])
    batch_1_xs = split_data(train_1_x, "2:3:4:1", opts["batch_size"])
    batch_1_ys = split_data(train_1_y, "2:3:4:1", opts["batch_size"])
    batch_2_xs = split_data(train_2_x, "3:4:1:2", opts["batch_size"])
    batch_2_ys = split_data(train_2_y, "3:4:1:2", opts["batch_size"])
    batch_3_xs = split_data(train_3_x, "4:1:2:3", opts["batch_size"])
    batch_3_ys = split_data(train_3_y, "4:1:2:3", opts["batch_size"])

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

    num_test_samples = len(test_x)
    opts["checkpoints_dir"] = "../checkpoints/plan_e/"
    opts["logs_dir"] = "../logs/plan_e/"
    model_name = "plan_e.ckpt"

    train_losses = []
    test_losses = []
    accuracy = []

    model = ModelV1(opts)
    model.build()

    num_samples = len(train_0_x)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        pos = 0
        end = len(batch_0_xs)
        for epoch in range(opts["epochs"]):

            if pos >= end:
                pos = 0

            while pos < end:

                batch_x = np.concatenate([batch_0_xs[pos], batch_1_xs[pos], batch_2_xs[pos], batch_3_xs[pos]], axis=0)
                batch_y = np.concatenate([batch_0_ys[pos], batch_1_ys[pos], batch_2_ys[pos], batch_3_ys[pos]], axis=0)

                feed_dict = {
                    model.inputs: batch_x,
                    model.labels: batch_y
                }
                fetches = [model.loss, model.train_step, model.global_step, model.merge_op]
                loss_value, _, global_step_value, merge_str = sess.run(
                    fetches, feed_dict
                )

                test_fetches = [model.loss, model.preds]
                test_feed_dict = {
                    model.inputs: test_x,
                    model.labels: test_y
                }
                test_loss_value, preds_value = sess.run(
                    test_fetches, test_feed_dict
                )
                test_losses.append(test_loss_value)

                preds_idx = np.argmax(preds_value, axis=1)
                labels_idx = np.argmax(test_y, axis=1)
                accuracy_ = float(np.sum(np.asarray(preds_idx == labels_idx)) / num_test_samples)
                accuracy.append(accuracy_)

                train_losses.append(loss_value)
                model.writer.add_summary(merge_str, global_step=global_step_value)

                if (epoch + 1) % 10 == 0:  # 逢9的epoch有存储操作，导致程序变慢
                    model.saver.save(sess, opts["checkpoints_dir"] + model_name, global_step=global_step_value)

                print("Epoch:%d, global_step:%d, train_loss:%.6f, test_loss:%.6f, test_accuracy:%.3f" % (
                    epoch, global_step_value, loss_value, test_loss_value, accuracy_))

                pos += 1

        model.writer.close()
        np.save("../results/plan_e/train_losses.npy", np.asarray(train_losses))
        np.save("../results/plan_e/test_losses.npy", np.asarray(test_losses))
        np.save("../results/plan_e/accuracy.npy", np.asarray(accuracy))


if __name__ == "__main__":
    train()
