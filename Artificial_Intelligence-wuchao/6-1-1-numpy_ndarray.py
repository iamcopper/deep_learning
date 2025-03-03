#!/usr/bin/env python
#
# 吴超《人工智能导论》：AI平台与系统（一）
# NumPy的数组类型：ndarray
#

import numpy as np

print("\n>>>>>> ndarray：")
# 创建ndarray
score = np.array([[80, 89, 86, 67, 79], [78, 97, 89, 67, 81]])

# 打印结果
print("score:\n", score)

print("\n>>>>>> ndarray属性：")

# 创建不同形状的数组
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([1, 2, 3, 4])
c = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6.0]]])

# 分别打印出形状
print("a.shape=", a.shape)
print("b.shape=", b.shape)
print("c.shape=", c.shape)

# 分别打印维数
print("a.ndim=", a.ndim)
print("b.ndim=", b.ndim)
print("c.ndim=", c.ndim)

# 分别打印数组元素数量
print("a.size=", a.size)
print("b.size=", b.size)
print("c.size=", c.size)

# 分别打印数组元素类型
print("a.dtype=", a.dtype)
print("b.dtype=", b.dtype)
print("c.dtype=", c.dtype)

print("\n>>>>>> ndarray的数据类型：")

# 创建数组时指定类型为 np.float32
a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

# 创建数组是未指定类型
b = np.array([[1, 2, 3], [4, 5, 6]])

# 打印数据类型
print("数组a: \n%s, \n数据类型: %s"%(a, a.dtype))
print("数组b: \n%s, \n数据类型: %s"%(b, b.dtype))