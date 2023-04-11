"""
@Project ：.ProjectCode 
@File    ：all
@Describe：所有的代码统合在一起
@Author  ：KlNon
@Date    ：2023/4/11 20:36 
"""
import os

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
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


# noinspection DuplicatedCode
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 56 * 56, 512)
        self.dropout = nn.Dropout(p=0.5)  # 加入一个50%的Dropout层
        self.fc2 = nn.Linear(512, 20)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 64 * 56 * 56)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# noinspection DuplicatedCode
net = Net()
# 加载数据集
train_dataset = datasets.ImageFolder('./assets/train', transform=train_transform)
test_dataset = datasets.ImageFolder('./assets/test', transform=test_transform)
train_loader = DataLoader(train_dataset, batch_size=30, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=30, shuffle=False, drop_last=True)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# 使用GPU运算,切换运算设备
device = torch.device('cpu')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)

# 模型位置
__PATH__ = './flower_net.pth'


# noinspection DuplicatedCode
def modelTest(test_loader, net):
    from main import device
    # 在测试集上评估模型
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            if device.type == 'cuda':
                images, labels = data[0].to(device), data[1].to(device)
            else:
                images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (
            100 * correct / total))


def modelTrain(train_loader, optimizer, net, criterion, path):
    from main import device
    epoch = 0
    while True:
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            if device.type == 'cuda':
                inputs, labels = data[0].to(device), data[1].to(device)
            else:
                inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch += 1
        if running_loss / len(train_loader) < 0.01 or epoch > 300:  # 设定一个终止条件，比如平均损失小于0.01
            break
        if epoch % 100 == 0:
            torch.save(net.state_dict(), path)
        print('Training finished after %d epochs' % epoch)
        print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, running_loss / len(train_loader)))


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    if os.path.exists(__PATH__):
        net.load_state_dict(torch.load(__PATH__))
        # net.eval()  # 将模型转为评估模式
    modelTrain(train_loader, optimizer, net, criterion, __PATH__)
    modelTest(test_loader, net)
    torch.save(net.state_dict(), __PATH__)
