#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def plot_activation_function(activation_function):
    """
    绘制激活函数
    :param activation_function: 激活函数名
    :return
    """

    x = np.arange(-10, 10, 0.1)
    y_activation_function = activation_function(x)
    
    # 绘制坐标轴
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data', 0))
    
    # 绘制曲线图
    plt.plot(x, y_activation_function)
    
    # 展示函数图像
    plt.show()

def sigmoid(x):
    """
    sigmoid函数
    :param x: np.array格式数据
    :return: sigmoid函数
    """
    return 1 / (1 + np.exp(-x))

def tanh(x):
    """
    tanh函数
    :param x: np.array格式数据
    :return: tanh函数
    """
    return ((np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)))

def relu(x):
    """
    relu函数
    :param x: np.array格式数据
    :return: relu函数
    """
    
    temp = np.zeros_like(x)
    if_bigger_zero = (x > temp)
    return x * if_bigger_zero

# 绘制函数图像
#plot_activation_function(sigmoid)
#plot_activation_function(tanh)
plot_activation_function(relu)