#!/usr/bin/env python

# 感知机模型用于分类
#  假设一个分类问题，我们有两种颜色的点分布在空间中
#  通过其中的感知机参数更新过程，我们可以了解感知机网络究竟是如何学习的。
#  这样不断重复，优化参数使得模型结果更加符合训练目标的过程，就被成为神经网络的学习过程。

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import numpy as np

X = np.array([[1, 1], [3, 3], [4, 3]])
y = [-1, 1, 1]

# 绘制散点图
for i, x in zip(y, X):
    if i == -1:
        plt.scatter(x[0], x[1], c = 'r')
    else:
        plt.scatter(x[0], x[1], c = 'b')
    
# 展示函数图像
plt.show()

########################################################
# 定义感知机类
class Perceptron(object):
    # 初始化参数w和b
    def __init__(self, learning_rate=1):
        self.w = np.array([0, 0]).reshape((-1, 1))
        self.b = 0
    
    def sign(self, x):
        return -1 if x < 0 else +1

    def calculate(self, X):
        yH = np.matmul(X, self.w) + self.b
        return np.apply_along_axis(self.sign, 1, yH)
    
    def get_wrong(self, X, yH, Y):
        for x, yh, y in zip(X, yH, Y):
            if yh != y:
                return {'x': x, 'y': y}
        return None
    
    def fit(self, X, y):
        while True:
            yH = self.calculate(X)
            wrong = self.get_wrong(X, yH, y)
            print("Wrong Point {}", wrong)
            if not wrong:
                break
            self.w = self.w + (wrong['x'] * wrong['y']).reshape(-1, 3)
            self.b = self.b + wrong['y']
            print("update w to {} update b to {}".format(self.w, self.x))

per = Perceptron()
per.fit(X, y)