#!/usr/bin/env python

# Lasso回归如何减少过拟合
# 从运行结果，可以看到所有的参数都趋向于0。
# Lasso回归的效果，避免了过拟合，在训练数据上仅为0.0486。
import numpy as np
from sklearn.linear_model import Lasso

X1 = np.array([1.83, 1.77, 1.67, 1.75, 1.72, 1.71, 1.80])
X2 = [x**2 for x in X1]
X3 = [x**3 for x in X1]
X4 = [x**4 for x in X1]
X5 = [x**5 for x in X1]
X6 = [x**6 for x in X1]

y = np.array([74, 71, 78, 74, 69, 80, 77])

X = np.stack((X1, X2, X3, X4, X5, X6))
X = np.transpose(X)

clf = Lasso(alpha=0.1)
#clf = Lasso(alpha=0.04)
clf.fit(X, y)

print(clf.score(X, y))
print(clf.coef_)