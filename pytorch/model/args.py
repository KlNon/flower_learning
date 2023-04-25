"""
@Project ：.ProjectCode 
@File    ：args
@Describe：
@Author  ：KlNon
@Date    ：2023/4/13 19:54 
"""

# 模型位置
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

from pytorch.model.net.model_net import create_network

PATH = './flower_net_VGG19.pth'
gdrive_dir = 'E:/.GraduationProject/ProjectCode/checkpoint/'
batch_size = 40
learning_rate = 0.001
normalize_mean = np.array([0.485, 0.456, 0.406])
normalize_std = np.array([0.229, 0.224, 0.225])

# 使用GPU运算,切换运算设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f'Running on: {str(device).upper()}')

# Define hyperparameters
model_name = 'resnet50'
output_size = 103
hidden_layers = [1000]

model = create_network(model_name, output_size, hidden_layers)

# Move model to device
model.to(device)

# 数据预处理
data_transforms = {'train': transforms.Compose([
    transforms.RandomChoice([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(180),
    ]),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        normalize_mean,
        normalize_std)
]), 'valid': transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        normalize_mean,
        normalize_std)
]), 'test': transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        normalize_mean,
        normalize_std)
])}

# transforms to train data set

# transforms to valid data set

# Load the datasets with ImageFolder
image_datasets = {'train_data': datasets.ImageFolder('E:/.GraduationProject/ProjectCode/assets/train',
                                                     transform=data_transforms['train']),
                  'valid_data': datasets.ImageFolder('E:/.GraduationProject/ProjectCode/assets/valid',
                                                     transform=data_transforms['valid']),
                  'test_data': datasets.ImageFolder('E:/.GraduationProject/ProjectCode/assets/test',
                                                    transform=data_transforms['test'])}
# 加载数据集 训练集,验证集,测试集

# Using the image datasets and the transforms, define the dataloaders
dataloaders = {
    'train_data': torch.utils.data.DataLoader(image_datasets['train_data'], batch_size=batch_size, shuffle=True,
                                              num_workers=12),
    'valid_data': torch.utils.data.DataLoader(image_datasets['valid_data'], batch_size=batch_size, shuffle=True,
                                              num_workers=12),
    'test_data': torch.utils.data.DataLoader(image_datasets['test_data'], batch_size=batch_size, shuffle=True,
                                             num_workers=12)}

# print(f"Train data: {len(dataloaders['train_data'].dataset)} images / {len(dataloaders['train_data'])} batches")
# print(f"Valid data: {len(dataloaders['valid_data'].dataset)} images / {len(dataloaders['valid_data'])} batches")
# print(f"Test  data: {len(dataloaders['test_data'].dataset)} images / {len(dataloaders['test_data'])} batches")

# 类别为训练集的类别
data_classes = image_datasets['train_data'].classes

