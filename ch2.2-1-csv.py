#!/usr/bin/env python

import os
import pandas as pd
import torch

os.makedirs(os.path.join('.', 'data'), exist_ok=True)
data_file = os.path.join('.', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')        # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

data = pd.read_csv(data_file)
print(data)

c1, c2, c3 = data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2]
print("-----------------------------------")
print(c1)
print(c2)
print(c3)

print("-----------------------------------")
c1 = c1.fillna(c1.mean())
print(c1)
c2 = pd.get_dummies(c2, dummy_na=True)
print(c2)
print(c3)

print("-----------------------------------")
x = torch.tensor(c1.to_numpy(dtype=float))
y = torch.tensor(c2.to_numpy(dtype=bool))
z = torch.tensor(c3.to_numpy(dtype=float))
print(x)
print(y)
print(z)