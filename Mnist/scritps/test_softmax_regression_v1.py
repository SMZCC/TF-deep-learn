# coding=utf-8
# date: 2018/12/12, 18:45
# name: smz

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from Mnist.configrations.options import *
from Mnist.modules.model import MnistSoftmax
from tensorflow.examples.tutorials.mnist import input_data



class Viualizer(object):
    def __init__(self, image):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.set_axis_off()
        self.im = self.ax.imshow(image)

    def show(self, image, label):
        """显示一张图，以及其对应的名字"""
        self.im.set_data(image)
        self.ax.set_title(label, fontdict={"fontsize":10})



def test():
    model = MnistSoftmax()
    model.build(LOGS_DIR)

    with tf.Session() as sess:

        model.saver.restore(sess, CHECKPOINTS_DIR+"-%d"%(13750))
        mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)
        total_images = len(mnist.test.images)
        image_one = np.reshape(mnist.test.images[0], (28, 28))
        visualizer = Viualizer(image_one)
        correct_num = 0
        for idx, image in enumerate(mnist.test.images):
            image = np.expand_dims(image, 0)
            label = np.expand_dims(mnist.test.labels[idx], 0)

            input_dict = {model.input:image}
            pred = sess.run(tf.argmax(model.pred, 1), feed_dict=input_dict)   # 0轴表示batch_size,1轴才是一个样本的结果
            label_value = sess.run(tf.argmax(model.label, 1), feed_dict={model.label:label})

            image_to_show = np.reshape(image, (28, 28))
            visualizer.show(image_to_show, "pred:%d, label:%d"%(pred[0], label_value[0]))
            plt.pause(1)
            iscorrect = sess.run(tf.equal(pred, label_value))
            if iscorrect:
                correct_num += 1

if __name__ == "__main__":
    test()


