import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from pytorch.model.net.model_net import Net

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
