# coding=utf-8
# date: 2018/12/11, 14:32
# name: smz

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

from Mnist.configrations.options import *
from Mnist.modules.model import MnistSoftmax

def read_mnist(data_dir):
    return input_data.read_data_sets(data_dir, one_hot=True)


def show_mnist(Mnist):
    """<Mnist>show"""
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    img = Mnist.train.images[0]             # images_shape: (55000, 784)
    img = img.reshape(-1, 28)
    ax.axis("off")
    ax.imshow(img)

    plt.show()

def MnistSoftmax_train(logs_dir, checkpoints_dir, Mnist):
    model = MnistSoftmax()
    model.build(logs_dir)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()

        sess.run(init)

        for epoch in range(EPOCHS):
            start = 0
            next_ = 0
            total_samples = len(Mnist.train.images)

            while next_ < total_samples:
                next_ = start + BATCH_SIZE
                inputs = Mnist.train.images[start:next_]
                labels = Mnist.train.labels[start:next_]
                input_dict = {model.input:inputs, model.label:labels}

                loss, _, merge_op_str, global_step = sess.run(fetches=[model.loss, model.train_step,
                                                                       model.merge_op, model.global_step],
                                                              feed_dict=input_dict)
                start = next_
                model.writer.add_summary(merge_op_str, global_step)
                print("epoch:%d, step:%d, loss:%.6f"%(epoch, global_step, loss))
            if (epoch + 1) % 5 == 0:
                model.saver.save(sess, checkpoints_dir, global_step=model.global_step)

        model.writer.close()



if __name__ == "__main__":

    mnist = read_mnist(DATA_DIR)
    # show_mnist(mnist)
    MnistSoftmax_train(LOGS_DIR, CHECKPOINTS_DIR, mnist)

