#!/usr/bin/env python

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# 搭建CNN神经网络模型
# 定义一个神经网络模型类，继承于nn.Module, 所有的神经网络都要继承nn.Module
class Net(nn.Module):
    # 初始化，构造四层卷积神经网络
    def __init__(self):
        super(Net, self).__init__()
        # 构造第一个二维卷积层
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        #self.bn1 = nn.
        # 构造第二个二维卷积层
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        # 构造第一个全连接层
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        # 构造第二个全连接层
        self.fc2 = nn.Linear(500, 10)

    # 所有继承nn.Module必须要实现的一个接口，该接口用来做前序网络的搭建。
    def forward(self, x):
        # 先将输入x经过第一个卷积层（以及一个激活函数ReLU）
        x = F.relu(self.conv1(x))
        # 再经过一个2x2的最大池化
        x = F.max_pool2d(x, 2, 2)
        # 再经过第二个卷积层（以及一个激活函数ReLU）
        x = F.relu(self.conv2(x))
        # 再经过一个2x2的最大池化
        x = F.max_pool2d(x, 2, 2)
        # 把它拉平（改变张量的尺寸，把张量做形状的变化，变成1维）
        x = x.view(-1, 4 * 4 * 50)
        # 再经过第一层全连接（以及一个激活函数ReLU）
        x = F.relu(self.fc1(x))
        # 再经过第二层全连接
        x = self.fc2(x)
        # 最后经过一层softmax（归一化）后输出
        return F.log_softmax(x, dim=1)

# 模型训练
def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # 清零上一个循环得到的梯度
        optimizer.zero_grad()
        # 给定数据，获取模型的输出（前序过程）
        output = model(data)
        # 获取损失函数结果
        loss = F.nll_loss(output, target)
        # 利用pytorch框架自动求导功能 对 损失函数求导
        loss.backward()
        # 使用优化器优化
        optimizer.step()
        if batch_idx % 1000 == 0:
            # 输出训练数据的准确率
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))

# 模型测试
def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    # 测试时，不要对网络进行更新(更新梯度)
    with torch.no_grad():
        for data, target in test_loader:
            # 计算模型针对测试数据的预测值
            output = model(data)
            # 计算模型预测的损失函数值
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # 对损失值求和，以得到整个模型的损失值
            pred = output.argmax(dim=1, keepdim=True)  # get the index of
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    
    print('\nTest set: Averaget loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100.*correct/len(test_loader.dataset)))

def main():
    # 一个BATCH有多少张图片
    BATCH_SIZE = 128
    # 要训练多少轮
    epoches = 5
    # 学习率
    lr = 1e-4
    
    torch.manual_seed(1)
    kwargs = {}
    # 获得训练数据
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size = BATCH_SIZE, shuffle=True, **kwargs
    )
    # 获得测试数据
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size = BATCH_SIZE, shuffle=True, **kwargs
    )

    # 获得模型
    model = Net()
    # 获得模型优化器（随机梯度下降）
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    for epoch in range(1, epoches + 1):
        train(model, train_loader, optimizer, epoch)
        test(model, test_loader)
    
    # 将模型保存成模型文件
    torch.save(model.state_dict(), "mnist_cnn.pt")

main()