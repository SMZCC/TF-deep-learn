# coding=utf-8
# date: 2019/1/17, 20:55
# name: smz

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

from FNN_skills.modules.model import ModelV1
from FNN_skills.configration.options import opts


def train():
    """plan_a:每个class依次取出一个batch的量进行训练
    batch_size=100
    一次，一个数据分布
    """
    train_0_x = np.load("../data/class_0_train_X.npy")
    train_0_y = np.load("../data/class_0_train_Y.npy")
    train_1_x = np.load("../data/class_1_train_X.npy")
    train_1_y = np.load("../data/class_1_train_Y.npy")
    train_2_x = np.load("../data/class_2_train_X.npy")
    train_2_y = np.load("../data/class_2_train_Y.npy")
    train_3_x = np.load("../data/class_3_train_X.npy")
    train_3_y = np.load("../data/class_3_train_y.npy")

    train_x = [train_0_x, train_1_x, train_2_x, train_3_x]
    train_y = [train_0_y, train_1_y, train_2_y, train_3_y]

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
    opts["checkpoints_dir"] = "../checkpoints/plan_a/"
    opts["logs_dir"] = "../logs/plan_a/"
    model_name = "plan_a.ckpt"
    train_losses = []
    test_losses = []
    accuracy = []

    model = ModelV1(opts)
    model.build()

    num_samples = len(train_0_x)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        for epoch in range(opts["epochs"]):
            start_pointer = 0
            while start_pointer + opts["batch_size"] <= num_samples:
                data_idx = 0
                end_pointer = start_pointer + opts["batch_size"]
                while data_idx < 4:
                    x_i = train_x[data_idx]
                    batch_x = x_i[start_pointer:end_pointer, :]
                    batch_y = train_y[data_idx][start_pointer:end_pointer, :]
                    data_idx += 1

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

                    if (epoch + 1) % 10 == 0:
                        model.saver.save(sess, opts["checkpoints_dir"] + model_name, global_step=global_step_value)

                    print("Epoch:%d, global_step:%d, train_x:%d, train_loss:%.6f, test_loss:%.6f, test_accuracy:%.3f" % (
                        epoch, global_step_value, data_idx, loss_value, test_loss_value, accuracy_))

                start_pointer = end_pointer

        model.writer.close()
        np.save("../results/plan_a/train_losses.npy", np.asarray(train_losses))
        np.save("../results/plan_a/test_losses.npy", np.asarray(test_losses))
        np.save("../results/plan_a/accuracy.npy", np.asarray(accuracy))


if __name__ == "__main__":
    train()








