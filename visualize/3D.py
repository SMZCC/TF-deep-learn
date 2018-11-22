# coding=utf-8
# date: 2018/11/16, 15:39
# name: smz

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker  # 控制坐标轴的显示精度



def show_data():

    data = np.array([[87.87, 88.63, 88.63, 89.01, 89.01], [87.87, 88.63, 88.63, 89.01, 89.01], [87.87, 88.63, 88.63, 89.01, 88.25], [87.87, 88.63, 88.63, 89.01, 89.01], [87.87, 88.63, 88.63, 89.01, 89.01]])

    # xs, ys = np.meshgrid([0.1, 0.01, 0.001, 0.0001, 0.00001], [0.1, 0.01, 0.001, 0.0001, 0.00001])
    xs, ys = np.meshgrid(range(5), range(5))
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel('lambda1')
    # ax.set_xticks([1e1, 1e2, 1e3, 1e4, 1e5])
    ax.set_xticklabels(['1e-1', '1e-2', '1e-3', '1e-4', '1e-5'])
    # ax.set_yticks()
    _sur = ax.plot_surface(xs, ys, data, cmap='rainbow')
    # plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.6f'))
    # ax.xaxis.set_label([0.1, 0.01, 0.001, 0.0001, 0.00001])
    plt.colorbar(_sur)
    plt.show()


if __name__ == '__main__':
    show_data()