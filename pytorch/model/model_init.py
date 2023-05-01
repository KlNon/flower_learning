"""
@Project ：.ProjectCode 
@File    ：model_init
@Describe：模型初始化
@Author  ：KlNon
@Date    ：2023/4/13 19:54 
"""

# 模型位置
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn

from torch.utils.data import DataLoader

from pytorch.model.net.model_net import create_network


# 保存辨别种类的模型时,存在checkpoint,病虫害的模型放在checkpoint1
def initialize_model(model_name='resnet50', which_file='Kind', which_model='checkpoint',
                     checkpoint_base='E:/.GraduationProject/ProjectCode/',
                     data_dir_base='E:/.GraduationProject/ProjectCode/assets/', batch_size=40, num_worker=4,
                     output_size=103, hidden_layers=None, return_params=None):
    if hidden_layers is None:
        hidden_layers = [1000]
    if return_params is None:
        return_params = ['model_name', 'output_size', 'hidden_layers', 'checkpoint_dir', 'data_dir', 'device', 'model',
                         'data_transforms', 'image_datasets', 'dataloaders', 'data_classes']

    # PATH = './flower_net_VGG19.pth'
    normalize_mean = np.array([0.485, 0.456, 0.406])
    normalize_std = np.array([0.229, 0.224, 0.225])

    checkpoint_dir = checkpoint_base + which_model + '/'
    data_dir = data_dir_base + which_file + '/'

    # 使用GPU运算,切换运算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f'Running on: {str(device).upper()}')

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
    image_datasets = {'train_data': datasets.ImageFolder(data_dir + 'train',
                                                         transform=data_transforms['train']),
                      'valid_data': datasets.ImageFolder(data_dir + 'valid',
                                                         transform=data_transforms['valid']),
                      'test_data': datasets.ImageFolder(data_dir + 'test',
                                                        transform=data_transforms['test'])}
    # 加载数据集 训练集,验证集,测试集

    # Using the image datasets and the transforms, define the dataloaders
    dataloaders = {
        'train_data': torch.utils.data.DataLoader(image_datasets['train_data'], batch_size=batch_size, shuffle=True,
                                                  num_workers=num_worker),
        'valid_data': torch.utils.data.DataLoader(image_datasets['valid_data'], batch_size=batch_size, shuffle=True,
                                                  num_workers=num_worker),
        'test_data': torch.utils.data.DataLoader(image_datasets['test_data'], batch_size=batch_size, shuffle=True,
                                                 num_workers=num_worker)}

    # print(f"Train data: {len(dataloaders['train_data'].dataset)} images / {len(dataloaders['train_data'])} batches")
    # print(f"Valid data: {len(dataloaders['valid_data'].dataset)} images / {len(dataloaders['valid_data'])} batches")
    # print(f"Test  data: {len(dataloaders['test_data'].dataset)} images / {len(dataloaders['test_data'])} batches")

    # 类别为训练集的类别
    data_classes = image_datasets['train_data'].classes

    all_params = {
        'model_name': model_name,
        'output_size': output_size,
        'hidden_layers': hidden_layers,
        'checkpoint_dir': checkpoint_dir,
        'data_dir': data_dir,
        'device': device,
        'model': model,
        'data_transforms': data_transforms,
        'image_datasets': image_datasets,
        'dataloaders': dataloaders,
        'data_classes': data_classes
    }
    return tuple(all_params[name] for name in return_params)


def init_cri_opti(model):
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, momentum=0.9)
    return criterion, optimizer


def freeze_parameters(root, freeze=True):
    [param.requires_grad_(not freeze) for param in root.parameters()]


# Save the checkpoint
def save_checkpoint(model_name, output_size, hidden_layers, model, class_to_idx, cat_label_to_name,
                    checkpoint_path='checkpoint.pt'):
    model.to('cpu')
    checkpoint = {'model_name': model_name,
                  'output_size': output_size,
                  'hidden_layers': hidden_layers,
                  'model_state_dict': model.state_dict(),
                  'class_to_idx': class_to_idx,
                  'cat_label_to_name': cat_label_to_name}

    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path='checkpoint.pt'):
    checkpoint = torch.load(checkpoint_path)

    model_name = checkpoint['model_name']
    output_size = checkpoint['output_size']
    hidden_layers = checkpoint['hidden_layers']

    model = create_network(model_name=model_name,
                           output_size=output_size, hidden_layers=hidden_layers)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.cat_label_to_name = checkpoint['cat_label_to_name']

    return model
