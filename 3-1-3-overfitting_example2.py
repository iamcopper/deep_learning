#!/usr/bin/env python

# 过拟合演示2：用身高X，预测寿命y
# 非线性模型：y = ax^6 + bx^5 + cx^4 + dx^3 + ex^2 + fx + g
#   从结果可以看出，该高阶非线性模型对训练数据拟合效果非常好，准确率高达0.999（过拟合）。
#   但如果换一个模型之前从没有见过的数据来预测，可以看到预测结果非常不准确，此为过拟合。
import numpy as np
from sklearn.linear_model import LinearRegression

X1 = np.array([1.83, 1.77, 1.67, 1.75, 1.72, 1.71, 1.80])
X2 = [x**2 for x in X1]
X3 = [x**3 for x in X1]
X4 = [x**4 for x in X1]
X5 = [x**5 for x in X1]
X6 = [x**6 for x in X1]

y = np.array([74, 71, 78, 74, 69, 80, 77])

X = np.stack((X1, X2, X3, X4, X5, X6))
X = np.transpose(X)
reg = LinearRegression().fit(X, y)

print(reg.score(X, y))