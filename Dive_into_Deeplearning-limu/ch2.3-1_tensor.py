#!/usr/bin/env python

import torch

# scalar
print("\n************************ 2.3.1 scalar ************************")
x = torch.tensor(3.0)
y = torch.tensor(2.0)
print(x + y)
print(x * y)
print(x / y)
print(x ** y)

# vector
print("\n************************ 2.3.2 vector ************************")
x = torch.arange(4)
print(x)
print(x[3])
print(len(x))
print(x.shape)

# matrix
print("\n************************ 2.3.3 matrix ************************")
A = torch.arange(20).reshape(5, 4)
print(A)
print(A.T)
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print(B)
print(B == B.T)

print("\n************************ 2.3.4 tensor ************************")
X = torch.arange(24).reshape(2, 3, 4)
print(X)

print("\n************************ 2.3.5 tensor operation ************************")
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()
print(A)
print(A + B)
print(A * B)

a = 2
X = torch.arange(24).reshape(2, 3, 4)
print(X)
print(X.shape)
print(a + X)
print((a + X).shape)

print("\n************************ 2.3.6 dimensionality reduction ************************")
x = torch.arange(4, dtype=torch.float32)
print(x)
print(x.sum())

print("\n>>> A(5,4)")
print(A)
print(A.shape)
print(A.sum())
print(A.numel())
print("\n>>> A.sum(axis=0) --- 降维，沿第0轴（行）求和")
A_sum_axis0 = A.sum(axis=0)
print(A_sum_axis0)
print(A_sum_axis0.shape)
print("\n>>> A.sum(axis=1) --- 降维，沿第1轴（列）求和")
A_sum_axis1 = A.sum(axis=1)
print(A_sum_axis1)
print(A_sum_axis1.shape)
print("\n>>>  沿着行和列求和，等价于对矩阵所有元素求和")
print(A.sum(axis=[0, 1]))
print("\n>>>  求矩阵中所有元素平均值")
print(A.mean())
print(A.sum()/A.numel())