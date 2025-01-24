#!/usr/bin/env python

# 过拟合演示1：用身高X，预测寿命y
# 线性模型：y = ax + b
#   从结果可以看出，线性模型拟合效果比较差，对训练数据的准确率仅为0.061。
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1.83], [1.77], [1.67], [1.75], [1.72], [1.71], [1.80]])
y = np.array([74, 71, 78, 74, 69, 80, 77])

reg = LinearRegression().fit(X, y)

print(reg.score(X, y))