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
from torchvision.models import VGG19_Weights

from pytorch.model.args import *
from pytorch.model.net.model_net import Net

vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
net = Net(vgg)

net.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.0001, momentum=0.9)
