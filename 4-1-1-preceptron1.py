#!/usr/bin/env python

# 感知机模型用于判断
import numpy as np

def perceptron(x, w, threshold):
    """
    感知机模型
    :param x: 输入数据 np.array格式
    :param w: 权重 np.array格式，需要与x一一对应
    :param threshold: 阈值
    :return: 0或者1
    """
    x = np.array(x)
    w = np.array(w)
    
    # 计算信息加权综合
    y_sum = np.sum(w * x)
    
    # 大于阈值返回1，否则返回0
    return 1 if y_sum > threshold else 0

# 输入数据
x = np.array([1, 1, 4])
# 输入权重
w = np.array([0.5, 0.2, 0.3])
# 返回结果
y = perceptron(x, w, 0.8)
print(y)