"""
@Project ：.ProjectCode 
@File    ：__init__
@Describe：
@Author  ：KlNon
@Date    ：2023/4/11 21:38 
"""

import torch
from torchvision import models

import torch.nn as nn
import torch.optim as optim

from pytorch.model.args import *
from pytorch.model.net.model_net import Net

net = Net()
net.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
