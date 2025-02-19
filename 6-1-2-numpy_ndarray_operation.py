#!/usr/bin/env python
#
# 吴超《人工智能导论》：AI平台与系统（一）
# NumPy的基本操作
#

import numpy as np
from numpy import array

print("\n>>>>>> 生成全部元素值为0的数组: np.zeros()")
zeros = np.zeros([3, 4])
print("zeros:\n%s" % (zeros))

print("\n>>>>>> 生成全部元素值为1的数组: np.ones()")
ones = np.ones([3, 4])
print("ones:\n%s" % (ones))

print("\n>>>>>> 生成对角数组（对角线的地方是1，其余地方是0）: np.eye()")
eye = np.eye(3, 4)
print("eye:\n%s" % (eye))
print(">>>>>> 生成方阵对角数组：np.eye(3, 3)，可以简写为np.eye(3)")
eye1 = np.eye(3)
print("eye1:\n%s" % (eye1))

print("\n>>>>>> 通过现有数组去生成: np.array()")
a = [[1, 2, 3], [4, 5, 6]]
a1 = np.array(a)
print("a:\n%s" % (a))
print("a1:\n%s" % (a))

print("\n>>>>>> 生成等间隔的数组: np.linspace(), np.arange()")
a = np.linspace(0, 90, 10)
print("a:\n%s" % (a))
b = np.arange(0, 90, 10)
print("b:\n%s" % (b))

print("\n>>>>>> 矩阵形状修改: array.reshape()")
a = array([[0,  1,  2,  3,  4,  5],
           [10, 11, 12, 13, 14, 15],
           [20, 21, 22, 23, 24, 25],
           [30, 31, 32, 33, 34, 35]])
print("a: \n%s" % (a))
print("a.shape = ", a.shape)
b = a.reshape([3, 8])
print("b: \n%s" % (b))
print("b.shape = ", b.shape)
# -1： 表示通过自动计算得到此处的值
c = a.reshape([-1, 12])
print("c: \n%s" % (c))
print("c.shape = ", c.shape)
# 求矩阵的转置
d = a.T
print("d: \n%s" % (d))
print("d.shape = ", d.shape)

print("\n>>>>>> 矩阵元素数据类型修改: array.astype()")
arr1 = np.array([[[1, 2, 3], [4, 5, 6]], [[12, 3, 34], [5, 6, 7]]])
arr2 = arr1.astype(np.float32)
print("arr1.dtype = %s" % (arr1.dtype))
print("arr2.dtype = %s" % (arr2.dtype))

print("\n>>>>>> 数组元素去重：np.unique()")
arr1 = np.array([[1, 2, 3, 4], [3, 4, 5, 6]])
arr2 = np.unique(arr1)
print("arr1: \n%s" % (arr1))
print("arr2: \n%s" % (arr2))

print("\n>>>>>> 数组运算：")
a = np.array([1, 2])
print("a * 3 = %s" % (a * 3))
b = np.array([3, 4])
print("a * b = %s" % (a * b))

print(a.sum(axis=None))