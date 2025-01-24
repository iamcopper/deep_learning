#!/usr/bin/env python

# 数据：游乐场经营者提供天气情况（如晴、雨，多云）、温度高低、湿度大小、风力强弱等气象特点以及又可当天是否前往游乐场。
# 目标：预测游客是否来游乐场游玩。
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from math import log
import warnings
warnings.filterwarnings("ignore")

def calc_entropy(total_num, count_dict):
    """
    计算信息熵
    :param total_num: 总样本数，例如总的样本数是14
    :param count_dict: 每类样本及其对应数目的字典，例如：{'前往游乐场':9, '不前往游乐场':5}
    :return: 信息熵
    """
    #初始化 ent 为0
    ent = 0
    # 对于每个类别
    for n in count_dict.values():
        # 如果属于该类别的样本数大于0
        if n > 0:
            # 计算概率
            p = n / total_num
            # 计算信息熵
            ent += -p * log(p, 2)
    # 返回信息熵
    return round(ent, 3)


# 原始数据
datasets = [
    ['晴'  , 29,  85,  '否', '0'],
    ['晴'  , 26,  88,  '是', '0'],
    ['多云', 29,  78,  '否', '1'],
    ['雨'  , 21,  96,  '否', '1'],
    ['雨'  , 20,  80,  '否', '1'],
    ['雨'  , 18,  70,  '是', '0'],
    ['多云', 18 , 65,  '是', '1'],
    ['晴'  , 22,  90,  '否', '0'],
    ['晴'  , 21,  68,  '否', '1'],
    ['雨'  , 24,  80,  '否', '1'],
    ['雨'  , 24,  83,  '是', '1'],
    ['多云', 22,  96,  '是', '1'],
    ['多云', 27,  75,  '否', '1'],
    ['晴'  , 21,  80,  '是', '0']
]

# 数据的列名
labels = ['天气', '温度', '湿度', '是否有风', '是否前往游乐场']

# 将温度大小分为大于26和小于等于26 这两个属性值
# 将湿度大小分为大于75和小于等于75 这两个属性值
for i in range(len(datasets)):
    if datasets[i][1] > 26:
        datasets[i][1] = '>26'
    else:
        datasets[i][1] = '<=26'
    if datasets[i][2] > 75:
        datasets[i][2] = '>75'
    else:
        datasets[i][2] = '<=75'

# 构建dataframe并查看数据
df = pd.DataFrame(datasets, columns=labels)
print(df)