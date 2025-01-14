#!/usr/bin/env python

import torch

X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
print(X)
print(X[0])
print(X[1])
print(X[2])
print(X[-1])
print(X[1:3])

print(X[:, 0])
print(X[:, 1])
print(X[:, 2])
print(X[:, 3])

X[1, 2] = 9
print(X)

X[0:2, :] = 12
print(X)