#!/usr/bin/env python3

import torch

x = torch.arange(12)
print(x)
print(x.shape)
print(x.numel())

y = x.reshape(3, 4)
print(y)
print(y.shape)
print(y.numel())

torch.zeros((2, 3, 4))

torch.ones((2, 3, 4))

torch.randn(3, 4)