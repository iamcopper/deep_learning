#!/usr/bin/env python

import numpy as np

# 创建ndarray
score = np.array([[80, 89, 86, 67, 79], [78, 97, 89, 67, 81]])

# 打印结果
print("score:\n", score)

#################################################################

# 创建不同形状的数组
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([1, 2, 3, 4])
c = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6.0]]])

# 分别打印出形状
print("a.shape=", a.shape)
print("b.shape=", b.shape)
print("c.shape=", c.shape)