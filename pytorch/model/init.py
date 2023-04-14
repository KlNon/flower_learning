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

net = models.vgg16()
net.classifier = nn.Sequential(nn.Linear(25088, 4096),  # vgg16
                               nn.ReLU(),
                               nn.Dropout(p=0.5),
                               nn.Linear(4096, 4096),
                               nn.ReLU(),
                               nn.Dropout(p=0.5),
                               nn.Linear(4096, 8))

net.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
