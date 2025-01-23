#!/usr/bin/env python
#
# 吴超《人工智能导论》：机器学学习经典模型（二）
# 
# 线性模型：逻辑回归模型解决分类问题演示
# 例子：使用逻辑回归模型，根据胸围、腰围判断男生、女生
# x1：胸围
# x2：腰围
# y：0代表女生，1代表男生

import numpy as np
# 引入逻辑回归模型
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# 准备数据
X = np.array([[82, 62], [79, 59], [81, 63], [72, 68], [73, 70], [74, 68]])
y = np.array([0, 0, 0, 1, 1, 1])
print(X.shape)

# 模型训练
clf = LogisticRegression(random_state=0).fit(X, y)

# 模型预测
print(clf.predict(np.array([[76, 72]])))
# 模型评价
print(clf.score(X, y))

# 绘图
plt.scatter(X[:, 0], X[:, 1], color='black')
plt.show()