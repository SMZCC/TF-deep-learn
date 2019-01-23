# coding=utf-8
# date: 2019/1/18, 15:56
# name: smz

import numpy as np
import matplotlib.pyplot as plt


def plot_plan_a_resutls():
    fig1 = plt.figure()
    ax = fig1.add_subplot(1, 1, 1)
    train_loss = np.load("../results/plan_a/train_losses.npy")
    test_loss = np.load("../results/plan_a/test_losses.npy")
    xs = range(len(train_loss))
    ax.plot(xs[:2500], train_loss[:2500], 'r-', label="train_loss")
    ax.plot(xs[:2500], test_loss[:2500], 'b.-', label="test_loss")
    plt.legend()
    plt.show()


def plot_plan_b_resutls():
    fig1 = plt.figure()
    ax = fig1.add_subplot(1, 1, 1)
    train_loss = np.load("../results/plan_b/train_losses.npy")
    test_loss = np.load("../results/plan_b/test_losses.npy")
    xs = range(len(train_loss))
    ax.plot(xs[:2500], train_loss[:2500], 'r-', label="train_loss")
    ax.plot(xs[:2500], test_loss[:2500], 'b--', label="test_loss")
    plt.legend()
    plt.show()


def plot_plan_c_results():
    fig1 = plt.figure()
    ax = fig1.add_subplot(1, 1, 1)
    train_loss = np.load("../results/plan_c/train_losses.npy")
    test_loss = np.load("../results/plan_c/test_losses.npy")
    xs = range(len(train_loss))
    ax.plot(xs[:2500], train_loss[:2500], 'r-', label="train_loss")
    ax.plot(xs[:2500], test_loss[:2500], 'b--', label="test_loss")
    plt.legend()
    plt.show()


def plot_plan_d_results():
    fig1 = plt.figure()
    ax = fig1.add_subplot(1, 1, 1)
    train_loss = np.load("../results/plan_d/train_losses.npy")
    test_loss = np.load("../results/plan_d/test_losses.npy")
    xs = range(len(train_loss))
    ax.plot(xs[:2500], train_loss[:2500], 'r-', label="train_loss")
    ax.plot(xs[:2500], test_loss[:2500], 'b--', label="test_loss")
    plt.legend()
    plt.show()


def plot_plan_e_results():
    fig1 = plt.figure()
    ax = fig1.add_subplot(2, 1, 1)
    train_loss = np.load("../results/plan_e/train_losses.npy")
    test_loss = np.load("../results/plan_e/test_losses.npy")
    xs = range(len(train_loss))
    ax.plot(xs[:2500], train_loss[:2500], 'r-', label="train_loss")
    ax.plot(xs[:2500], test_loss[:2500], 'b--', label="test_loss")
    ax.set_title("losses")

    ax_accuracy = fig1.add_subplot(2, 1, 2)
    accuracy = np.load("../results/plan_e/accuracy.npy")
    steps = range(len(accuracy))

    ax_accuracy.plot(steps, accuracy, 'r-', label="train_accuracy")
    ax_accuracy.set_title("accuracy")

    plt.legend()
    plt.show()


def plot_all_losses(show_num):
    """

    :param show_num: 显示几次迭代的结果
    :return:
    """
    fig = plt.figure()
    ax_loss = fig.add_subplot(3, 1, 1)
    plan_a_train_loss = np.load("../results/plan_a/train_losses.npy")
    plan_b_train_loss = np.load("../results/plan_b/train_losses.npy")
    plan_c_train_loss = np.load("../results/plan_c/train_losses.npy")
    plan_d_train_loss = np.load("../results/plan_d/train_losses.npy")
    plan_e_train_loss = np.load("../results/plan_e/train_losses.npy")

    plan_a_test_loss = np.load("../results/plan_a/test_losses.npy")
    plan_b_test_loss = np.load("../results/plan_b/test_losses.npy")
    plan_c_test_loss = np.load("../results/plan_c/test_losses.npy")
    plan_d_test_loss = np.load("../results/plan_d/test_losses.npy")
    plan_e_test_loss = np.load("../results/plan_e/test_losses.npy")

    plan_a_accuracy = np.load("../results/plan_a/accuracy.npy")
    plan_b_accuracy = np.load("../results/plan_b/accuracy.npy")
    plan_c_accuracy = np.load("../results/plan_c/accuracy.npy")
    plan_d_accuracy = np.load("../results/plan_d/accuracy.npy")
    plan_e_accuracy = np.load("../results/plan_e/accuracy.npy")


    xs = range(show_num)
    ax_loss.plot(xs, plan_a_train_loss[:show_num], 'r-', label="plan_a")
    ax_loss.plot(xs, plan_b_train_loss[:show_num], 'b-', label="plan_b")
    ax_loss.plot(xs, plan_c_train_loss[:show_num], 'y-', label='plan_c')
    ax_loss.plot(xs, plan_d_train_loss[:show_num], 'g-', label="plan_d")
    ax_loss.plot(xs, plan_e_train_loss[:show_num], 'k-', label="plan_e")
    ax_loss.set_title("train_loss", fontdict={"fontsize":10})
    ax_loss.legend()

    ax_test_loss = fig.add_subplot(3, 1, 2)
    ax_test_loss.plot(xs, plan_a_test_loss[:show_num], 'r.-', label="plan_a")
    ax_test_loss.plot(xs, plan_b_test_loss[:show_num], 'b.-', label="plan_b")
    ax_test_loss.plot(xs, plan_c_test_loss[:show_num], 'y.-', label="plan_c")
    ax_test_loss.plot(xs, plan_d_test_loss[:show_num], 'g.-', label="plan_d")
    ax_test_loss.plot(xs, plan_e_test_loss[:show_num], 'k.-', label="plan_e")
    ax_test_loss.set_title("test_loss", fontdict={"fontsize":10})
    ax_test_loss.legend()

    ax_accuracy = fig.add_subplot(3, 1, 3)
    ax_accuracy.plot(xs, plan_a_accuracy[:show_num], 'r:', label="plan_a")
    ax_accuracy.plot(xs, plan_b_accuracy[:show_num], 'b:', label="plan_b")
    ax_accuracy.plot(xs, plan_c_accuracy[:show_num], 'y:', label="plan_c")
    ax_accuracy.plot(xs, plan_d_accuracy[:show_num], 'g:', label="plan_d")
    ax_accuracy.plot(xs, plan_e_accuracy[:show_num], 'k:', label="plan_e")
    ax_accuracy.set_title("accuracy", fontdict={"fontsize": 10})
    ax_accuracy.legend()

    plt.show()


if __name__ == "__main__":
    # plot_plan_a_resutls()
    # plot_plan_b_resutls()
    # plot_plan_c_results()
    # plot_plan_e_results()
    plot_all_losses(1000)
