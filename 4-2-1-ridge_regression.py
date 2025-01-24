#!/usr/bin/env python

# 岭回归(Ridge Regression)如何减少过拟合
# 同样的问题，在使用岭回归替换线性回归后，避免了过拟合问题。
# 该模型在训练数据上的准确率也仅为0.053。
import numpy as np
from sklearn.linear_model import Ridge

X1 = np.array([1.83, 1.77, 1.67, 1.75, 1.72, 1.71, 1.80])
X2 = [x**2 for x in X1]
X3 = [x**3 for x in X1]
X4 = [x**4 for x in X1]
X5 = [x**5 for x in X1]
X6 = [x**6 for x in X1]

y = np.array([74, 71, 78, 74, 69, 80, 77])

X = np.stack((X1, X2, X3, X4, X5, X6))
X = np.transpose(X)

clf = Ridge(alpha=1.0)
clf.fit(X, y)

print(clf.score(X, y))
print(clf.coef_)