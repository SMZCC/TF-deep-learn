# coding=utf-8
# date: 2019/1/18, 17:17
# name: smz


import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

from FNN_skills.modules.model import ModelV1
from FNN_skills.configration.options import opts


def train():
    """plan_d: batch_size=100
    1：2：3：4, 一份是10
    由于第4类一次取40个，使得循环会在4k多的时候结束
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
    opts["checkpoints_dir"] = "../checkpoints/plan_d/"
    opts["logs_dir"] = "../logs/plan_d/"
    model_name = "plan_d.ckpt"

    train_losses = []
    test_losses = []
    accuracy = []

    model = ModelV1(opts)
    model.build()

    num_samples = len(train_0_x)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        start_pointer_1 = 0
        start_pointer_2 = 0
        start_pointer_3 = 0
        start_pointer_4 = 0
        step_1 = 10
        step_2 = 20
        step_3 = 30
        step_4 = 40

        for epoch in range(opts["epochs"]):
            if start_pointer_1 + step_1 > num_samples:
                start_pointer_1 = 0
            if start_pointer_2 + step_2 > num_samples:
                start_pointer_2 = 0
            if start_pointer_3 + step_3 > num_samples:
                start_pointer_3 = 0
            if start_pointer_4 + step_4 > num_samples:
                start_pointer_4 = 0

            while start_pointer_1 + step_1 <= num_samples and start_pointer_2 + step_2 <= num_samples \
                and start_pointer_3 + step_3 <= num_samples and start_pointer_4 + step_4 <= num_samples:

                end_pointer_1 = start_pointer_1 + step_1
                end_pointer_2 = start_pointer_2 + step_2
                end_pointer_3 = start_pointer_3 + step_3
                end_pointer_4 = start_pointer_4 + step_4

                batch_x_0 = train_x[0][start_pointer_1:end_pointer_1]
                batch_x_1 = train_x[1][start_pointer_2:end_pointer_2]
                batch_x_2 = train_x[2][start_pointer_3:end_pointer_3]
                batch_x_3 = train_x[3][start_pointer_4:end_pointer_4]
                batch_x = np.concatenate([batch_x_0, batch_x_1, batch_x_2, batch_x_3], axis=0)

                batch_y_0 = train_y[0][start_pointer_1:end_pointer_1]
                batch_y_1 = train_y[1][start_pointer_2:end_pointer_2]
                batch_y_2 = train_y[2][start_pointer_3:end_pointer_3]
                batch_y_3 = train_y[3][start_pointer_4:end_pointer_4]
                batch_y = np.concatenate([batch_y_0, batch_y_1, batch_y_2, batch_y_3], axis=0)

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

                print("Epoch:%d, global_step:%d, train_loss:%.6f, test_loss:%.6f, test_accuracy:%.3f" % (
                    epoch, global_step_value, loss_value, test_loss_value, accuracy_))

                start_pointer_1 = end_pointer_1
                start_pointer_2 = end_pointer_2
                start_pointer_3 = end_pointer_3
                start_pointer_4 = end_pointer_4

        model.writer.close()
        np.save("../results/plan_d/train_losses.npy", np.asarray(train_losses))
        np.save("../results/plan_d/test_losses.npy", np.asarray(test_losses))
        np.save("../results/plan_d/accuracy.npy", np.asarray(accuracy))


if __name__ == "__main__":
    train()