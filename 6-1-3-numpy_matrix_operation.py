#!/usr/bin/env python
#
# 吴超《人工智能导论》：AI平台与系统（一）
# NumPy的矩阵运算(Matrix Operation)
#

import numpy as np

print("\n>>>>>> 矩阵的创建")
# 将ndarray转化为Matrix
a = np.array([[1, 2, 4],
              [2, 5, 3],
              [7, 8, 9]])
A = np.mat(a)
print("type(a)=", type(a))
print("type(A)=", type(A))
# 使用Matlab语法传入一个字符串来生成Matrix
B = np.mat('1, 2; 3, 4; 5, 6')
print("type(B)=", type(B))

print("\n>>>>>> 矩阵与向量的乘法（点乘）")
x = np.array([[1], [2], [3]])
X = np.mat(x)
print("type(A)=%s, A.shape=%s" % (type(A), A.shape))
print("type(x)=%s, x.shape=%s" % (type(x), x.shape))
Y = A*x
print(">>> Y=A*x")
print("type(Y)=%s" % (type(Y)))
print("Y=", Y)

print("\n>>>>>> 矩阵与矩阵的乘法（点乘）")
Y = A*B
print(">>> Y=A*B")
print("type(Y)=%s" % (type(Y)))
print("Y=", Y)

print("\n>>>>>> 矩阵的逆运算（相当于矩阵分之一）")
Y = A.I
print(">>> Y=A.I")
print("type(Y)=%s" % (type(Y)))
print("Y=", Y)

print("\n>>>>>> 矩阵的指数表示n个矩阵连乘")
Y = A ** 4
print(">>> Y=A**4")
print("type(Y)=%s" % (type(Y)))
print("Y=", Y)
print(Y.sum(axis=None))