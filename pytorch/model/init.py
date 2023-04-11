"""
@Project ：.ProjectCode 
@File    ：__init__
@Describe：
@Author  ：KlNon
@Date    ：2023/4/11 21:38 
"""

import torch

from pytorch.model.net.model_net import Net
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# 数据预处理
train_transform = transforms.Compose([
    transforms.RandomRotation(40),  # 随机旋转度数
    transforms.RandomHorizontalFlip(),  # 水平翻转
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),  # 将图像转换为Tensor
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.229, 0.224, 0.225)),  # 数据归一化
])

test_transform = transforms.Compose([
    transforms.RandomRotation(40),  # 随机旋转度数
    transforms.RandomHorizontalFlip(),  # 水平翻转
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),  # 将图像转换为Tensor
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.229, 0.224, 0.225)),  # 数据归一化
])


net = Net()
# 加载数据集
train_dataset = datasets.ImageFolder('./assets/train', transform=train_transform)
test_dataset = datasets.ImageFolder('./assets/test', transform=test_transform)
train_loader = DataLoader(train_dataset, batch_size=30, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=30, shuffle=False, drop_last=True)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.009, momentum=0.9)

# 使用GPU运算,切换运算设备
device = torch.device('cpu')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)

# 模型位置
__PATH__ = './flower_net.pth'

