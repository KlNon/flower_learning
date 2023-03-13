"""
@Project ：.ProjectCode 
@File    ：dataPreparation
@Describe：
@Author  ：KlNon
@Date    ：2023/3/13 16:11 
"""
import os
from PIL import Image
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms


# 数据清洗
# 遍历文件夹
# 获取文件路径
# 如果是图片文件，则打开
# 如果文件损坏,则删除
def clean_data(data_dir):
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        try:
            with Image.open(file_path) as img:
                img.verify()
        except:
            os.remove(file_path)
            print('Removed: ', file_path)


# 数据标准化
def normalize_data(dataset):
    # 定义标准化参数
    mean = [0.5, 0.5, 0.5]  # RGB通道的均值
    std = [0.5, 0.5, 0.5]  # RGB通道的标准差

    # 对图像进行标准化
    normalize = transforms.Normalize(mean=mean, std=std)

    # 对数据集进行标准化
    for i in range(len(dataset)):
        image, label = dataset[i]
        image = normalize(image)

        dataset[i] = (image, label)

    return dataset


# 特征提取
def extract_features(data_dir):
    # 定义数据转换
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    # 加载数据集
    dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)

    # 加载预训练模型
    model = models.resnet18(pretrained=True)
    model.eval()

    # 提取特征
    features = []
    labels = []
    with torch.no_grad():
        for images, targets in dataset:
            features.append(model(images.unsqueeze(0)).squeeze().numpy())
            labels.append(targets)

    return torch.tensor(features), torch.tensor(labels)


# 数据分割
def split_data(features, labels):
    dataset = torch.utils.data.TensorDataset(features, labels)

    # 定义数据集和分割比例
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size

    # 按照分割比例分割数据集
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 定义数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    return train_loader, test_loader


# 数据准备
def prepare_data(data_dir):
    # 数据清洗
    clean_data(data_dir)

    # 特征提取
    features, labels = extract_features(data_dir)

    # 数据分割
    train_loader, test_loader = split_data(features, labels)

    # 数据标准化
    train_loader = normalize_data(train_loader)
    test_loader = normalize_data(test_loader)

    return train_loader, test_loader

