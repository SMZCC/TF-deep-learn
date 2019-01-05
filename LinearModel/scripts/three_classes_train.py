# coding=utf-8
# date: 2019/1/1, 19:38
# name: smz

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from LinearModel.modules.model3 import ModelThreeClasses
from LinearModel.configuration.options import opts
from LinearModel.scripts.gen_data import generate_data


def gen_train_data():
    np.random.seed(10)
    fields_num = 2
    num_classes = 3
    sample_size = 2000
    mean = np.random.randn(fields_num)
    cov = np.eye(fields_num)
    diffs = [[3.0], [3.0, 0.0]]   # 第三类样本中心与第二类样本中心之间只有y方向上的误差,第二类样本与第一类样本在x和y方向上均偏移3.0
    train_X, train_Y = generate_data(num_classes=num_classes, sample_size=sample_size, mean=mean, cov=cov, diffs=diffs)
    np.save("../data/train_data_X3.npy", train_X)
    np.save("../data/train_data_Y3.npy", train_Y)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    colors = ['r' if np.argmax(label) == 0 else 'b' if np.argmax(label) == 1 else 'y' for label in train_Y]
    ax.scatter(train_X[:, 0], train_X[:, 1], c=colors)
    ax.set_xlabel("Scaled age(in years)")
    ax.set_ylabel("Tumor size(in cm)")

    plt.show()


def train_3_classes():
    """这个有问题，因为使用softmax表示的结果和使用sigmoid的那个模型是不同的,需要重写模型"""
    model3 = ModelThreeClasses(opts)
    model3.build()

    train_x3 = np.load("../data/train_data_X3.npy")
    train_y3 = np.load("../data/train_data_Y3.npy")

    model_name = "model3s.ckpt"

    num_samples = len(train_x3)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        for epoch in range(opts["epochs"]):
            start_pointer = 0
            train_x, train_y = shuffle(train_x3, train_y3)
            while start_pointer < num_samples:
                end_pointer = start_pointer + opts["batch_size"]
                batch_x = train_x[start_pointer:end_pointer]
                batch_y = train_y[start_pointer:end_pointer]
                start_pointer = end_pointer

                feed_dict = {model3.inputs: batch_x, model3.labels: batch_y}
                loss_value, glob_step_value, merge_str, _= sess.run(fetches=[model3.loss, model3.global_step, model3.merge_op,
                                                                                           model3.train_step],
                                                                                  feed_dict=feed_dict)
                model3.writer.add_summary(merge_str, global_step=glob_step_value)

                print("epoch:%d, step:%d, loss:%.6f"%(epoch, glob_step_value, loss_value))

            if (epoch + 1) % 10 == 0:
                model3.saver.save(sess, opts["checkpoints_dir"] + model_name, global_step=model3.global_step)



if __name__ == "__main__":
    # gen_train_data()
    train_3_classes()