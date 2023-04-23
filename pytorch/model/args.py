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

PATH = './flower_net_VGG19.pth'
batch_size = 40
learning_rate = 0.001

# 使用GPU运算,切换运算设备
device = torch.device('cpu')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 数据预处理

train_transform = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 加载数据集 训练集,验证集,测试集
train_dataset = datasets.ImageFolder('./assets/train', transform=train_transform)
val_dataset = datasets.ImageFolder('./assets/valid', transform=test_transform)
test_dataset = datasets.ImageFolder('./assets/test', transform=test_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# 类别为训练集的类别
data_classes = train_dataset.classes
