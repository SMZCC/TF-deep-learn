# coding=utf-8
# date: 2018/11/16, 15:39
# name: smz

import copy
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


def demo_colorful_surface():

    np.random.seed(10)

    x = np.linspace(0, 1, 20)
    y = np.linspace(0, 1, 20)
    xs, ys = np.meshgrid(x, y)

    z = np.exp(x**2+y**2)

    fig = plt.figure()
    # ax3d = Axes3D(fig)
    ax3d_1 = fig.add_subplot(121, projection='3d')
    colors_map_1 = np.random.rand(20, 20)
    colors_map_2 = np.random.rand(20, 20)
    colors_map_3 = np.random.rand(20, 20)
    r = np.random.rand(20, 20)

    colors_map = np.stack([colors_map_1, colors_map_2, colors_map_3, r], axis=2)

    ax_2_colors_map = copy.deepcopy(colors_map)
    ax3d_2 = fig.add_subplot(122, projection="3d")

    ax_2_colors_map[0:2, 0:2, :] = [[[0.1, 0.1, 0.1, 0.5], [0.1, 0.1, 0.1, 0.5]],
                               [[0.9, 0.9, 0.9, 0.5], [0.9, 0.9, 0.9, 0.5]]]

    surf_1 = ax3d_1.plot_surface(xs, ys, z, facecolors=colors_map)
    surf_2 = ax3d_2.plot_surface(xs, ys, z, facecolors=ax_2_colors_map)


    plt.show()


def demo_python():
    x = 100
    def hello():

        print(x)

    hello()



if __name__ == '__main__':
    demo_python()