#!/usr/bin/env python
#
# 吴超《人工智能导论》：机器学学习经典模型（一）
# 
# 线性回归模型的演示
# 例子：使用线性回归模型来根据房子与市中心的距离来预测房价
# 1. 使用sklearn库来创建线性回归模型；
# 2. 使用梯度下降算法来训练模型；

# import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import numpy as np
# 导入sklearn，选择线型回归模型
from sklearn.linear_model import LinearRegression

# 准备数据
# 房子与市中心的距离
X = np.array([[5], [10], [12], [13], [15]])
# 房子的价格
y = np.array([20, 15, 8, 6, 4])
print("X=", X)
print("y=", y)

# 模型训练：自动调用梯度下降算法，返回模型reg
reg = LinearRegression().fit(X, y)

# 线性回归模型（一元一次线性回归）：y = ax + b
# 打印模型斜率：a
print(reg.coef_)
# 打印模型截距：b
print(reg.intercept_)
# 评价模型：使用测试数据检验模型，对模型评分（1是最高）
print(reg.score(X, y))

# 试用模型进行预测
print(reg.predict(np.array([[3]])))

# 绘图：plt.scatter()绘制散点图
plt.scatter(X, y, color='black')
x_test = np.linspace(0, 15, 100)
y_test = reg.predict(x_test[:, np.newaxis])

# 绘图：plt.plot()绘制训练得到的模型函数曲线
plt.plot(x_test, y_test, color='blue', linewidth=3)
plt.show()