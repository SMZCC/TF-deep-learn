# coding=utf-8
# date: 2018/12/22, 13:48
# name: smz


import numpy as np
import matplotlib.pyplot as plt


def generate_data(num_classes, sample_size, mean, cov, diffs, one_hot=True):
    """
    样本： 年龄 肿瘤大小  良性/恶性
    args:
        sample_size： 样本的个数
        mean:  同类样本中，一个样本不同维度的均值
        cov:   同类样本中，一个样本不同纬度之间的协方差
        diff:  不同类样本之间的中心距离
        one_hot:  bool，指明标签是否使用one_hot编码表示,如果使用one_hot编码的话，分类的时候要使用softmax，
                  否则使用label*tf.log(outs) + (1-label)*tf.log(outs)
    :return:  X, Y : <tuple>, <tuple>
    """
    samples_per_class = int(sample_size / num_classes)

    x0 = np.random.multivariate_normal(mean, cov, size=samples_per_class)
    y0 = np.zeros(samples_per_class)    # 第一类样本的标签全是0

    for idx, diff in enumerate(diffs):   # 遍历其他类样本与第一类样本的距离
        x1 = np.random.multivariate_normal(mean + diff, cov, size=samples_per_class)
        y1 = (idx + 1) * np.ones(samples_per_class)

        x0 = np.concatenate((x0, x1))   # 用原来的数据类吸收新生成的数据类
        y0 = np.concatenate((y0, y1))

    if one_hot:
        class_idx = [ y0 == class_number for class_number in range(num_classes)]  # class_idx中两个元素：第一个元素为y0与0比较的结果，第二个元素为y0与1比较的结果
        class_idxs = [np.expand_dims(array, axis=1) for array in class_idx]
        y = np.asarray(np.hstack(class_idxs), dtype=np.float32)
        samples = zip(x0, y)
        samples = list(samples)
        np.random.shuffle(samples)

    else:
        samples = zip(x0, y0)
        samples = list(samples)
        np.random.shuffle(samples)

    X, Y = zip(*samples)
    return np.asarray(X), np.asarray(Y)




if __name__ == "__main__":
    np.random.seed(10)
    sample_size = 1000
    num_fields = 2
    mean = np.random.randn(num_fields)
    cov = np.eye(num_fields)     # 样本各维的协方差
    diffs = [3.0]  # 各类样本中心的距离

    X, Y = generate_data(num_fields, sample_size, mean, cov, diffs, one_hot=False)
    np.save("../data/train_data_X.npy", X)
    np.save("../data/train_data_Y.npy", Y)

    # colors = ['r' if label[0] == 1 else 'b' for label in Y]   # 1类样本显示为红色，二类样本显示为蓝色
    colors = ['r' if label == 1 else 'b' for label in Y]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(X[:, 0], X[:, 1], c=colors)
    plt.show()


