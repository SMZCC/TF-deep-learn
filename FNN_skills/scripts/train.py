# coding=utf-8
# date: 2019/1/15, 21:27
# name: smz

import numpy as np
import tensorflow as tf
from FNN_skills.modules.model import ModelV1
from FNN_skills.configration.options import opts


def train():
    train_x = np.load(opts["train_x"])
    train_y = np.load(opts["train_y"])

    num_samples = len(train_x)

    model_v1 = ModelV1(opts)
    model_v1.build()

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        for epoch in range(opts["epochs"]):
            start_pointer = 0
            while start_pointer < num_samples:
                end_pointer = start_pointer + opts["batch_size"]
                batch_x = train_x[start_pointer:end_pointer]
                batch_y = train_y[start_pointer:end_pointer]
                start_pointer = end_pointer
                feed_dict = {
                    model_v1.inputs:batch_x,
                    model_v1.labels:batch_y
                }
                fetches = [model_v1.loss, model_v1.merge_op, model_v1.train_step, model_v1.global_step]
                loss_value, merge_str, _, global_step_value = sess.run(fetches=fetches, feed_dict=feed_dict)
                print("Epoch:%d, step:%d, loss:%.6f"%(epoch, global_step_value, loss_value))
                model_v1.writer.add_summary(merge_str, global_step=global_step_value)
            if (epoch + 1) % 10 == 0:
                model_v1.saver.save(sess, opts["checkpoints_dir"]+opts["model_name"], global_step=model_v1.global_step)

    model_v1.writer.close()


if __name__ == "__main__":
    train()

