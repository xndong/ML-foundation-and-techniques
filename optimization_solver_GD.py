# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 23:49:31 2019

@author: DongXiaoning
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def cal_rosenbrock(x1, x2):
    """
    计算rosenbrock函数的值
    :param x1:
    :param x2:
    :return:
    """
    return (x1 - 3) ** 2 + (x2 + 1) ** 2


def cal_rosenbrock_prax(x1, x2):
    """
    对x1求偏导
    """
    return 2 * (x1 - 3)

def cal_rosenbrock_pray(x1, x2):
    """
    对x2求偏导
    """
    return 2 * (x2 + 1)

def for_rosenbrock_func(max_iter_count=100000, step_size=0.001):
    #pre_x = np.zeros((2,), dtype=np.float32)
    pre_x = np.array([-65,38], dtype=np.float32)
    loss = 10
    iter_count = 0

    x1 = np.arange(-5, 5, 0.005, dtype=np.float32)
    x2 = np.arange(-5, 5, 0.005, dtype=np.float32)
    x1, x2 = np.meshgrid(x1, x2)
    y = (x1 - 3) ** 2 + (x2 + 1) ** 2
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(x1, x2, y, cmap='rainbow')

    while loss > 0.001 and iter_count < max_iter_count:
        error = np.zeros((2,), dtype=np.float32)
        error[0] = cal_rosenbrock_prax(pre_x[0], pre_x[1])
        error[1] = cal_rosenbrock_pray(pre_x[0], pre_x[1])

        for j in range(2):
            pre_x[j] -= step_size * error[j]
        if iter_count % 250 == 0:
            ax.scatter(pre_x[0], pre_x[1],(pre_x[0] - 3) ** 2 + (pre_x[1] + 1) ** 2, marker='^', c='k')
        loss = cal_rosenbrock(pre_x[0], pre_x[1])  # 最小值为0

        print("error:", error, "pre_x:", pre_x, "iter_count: ", iter_count, "the loss:", loss)
        iter_count += 1
    plt.show()
    return pre_x


if __name__ == '__main__':
    w = for_rosenbrock_func()
    print(w)