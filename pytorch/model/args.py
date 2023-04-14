"""
@Project ：.ProjectCode 
@File    ：args
@Describe：
@Author  ：KlNon
@Date    ：2023/4/13 19:54 
"""

# 模型位置
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

PATH = './flower_net_VGG16.pth'
batch_size = 4
learning_rate = 0.001

# 使用GPU运算,切换运算设备
device = torch.device('cpu')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 数据预处理

train_transform = transforms.Compose([
    transforms.RandomRotation(40),  # 随机旋转度数
    transforms.RandomHorizontalFlip(),  # 水平翻转
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),  # 将图像转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[  # 图像归一化
        0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.RandomRotation(40),  # 随机旋转度数
    transforms.RandomHorizontalFlip(),  # 水平翻转
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),  # 将图像转换为Tensor
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.229, 0.224, 0.225)),  # 数据归一化
])

# 加载数据集 训练集,验证集,测试集
train_dataset = datasets.ImageFolder('./assets/train', transform=train_transform)
val_dataset = datasets.ImageFolder('./assets/val', transform=test_transform)
test_dataset = datasets.ImageFolder('./assets/test', transform=test_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# 类别为训练集的类别
data_classes = train_dataset.classes
