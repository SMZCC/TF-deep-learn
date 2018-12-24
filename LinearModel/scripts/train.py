# coding=utf-8
# date: 2018/12/24, 15:27
# name: smz

import numpy as np
import tensorflow as tf
from LinearModel.modules.model import TumorModel
from LinearModel.configuration.options import opts


def TumorModelTrain():
    data_X = np.load("../data/train_data_X.npy")
    data_Y = np.load("../data/train_data_Y.npy")
    data_Y = np.expand_dims(data_Y, axis=1)

    tumor_model = TumorModel(opts)
    tumor_model.build()
    num_samples = len(data_X)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        for epoch in range(opts["epochs"]):
            start_pointer = 0
            while start_pointer < num_samples:
                batch_X = data_X[start_pointer:start_pointer+opts["batch_size"]]
                batch_Y = data_Y[start_pointer:start_pointer+opts["batch_size"]]
                feed_dict = {tumor_model.inputs: batch_X, tumor_model.labels:batch_Y}

                loss_value, global_step_value, _, merge_string = sess.run(fetches=[tumor_model.loss, tumor_model.global_step,
                                                                                   tumor_model.train_step, tumor_model.merge_op],
                                                                          feed_dict=feed_dict)
                print("epoch:%d, step:%d, loss:%.6f"%(epoch, global_step_value, loss_value))
                start_pointer += start_pointer + opts["batch_size"]

                tumor_model.writer.add_summary(merge_string, global_step=global_step_value)

            if (epoch + 1) % 5 == 0:
                tumor_model.saver.save(sess, opts["checkpoints_dir"]+"tumor_model", global_step=tumor_model.global_step)


if __name__ == "__main__":
    TumorModelTrain()