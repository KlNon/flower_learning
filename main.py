import os

import torch

from pytorch.datasets.flower_dataset import *
from pytorch.model.net.model_net import Net
from pytorch.model.train.model_train import modelTrain
from pytorch.model.test.model_test import modelTest

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

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    if os.path.exists(__PATH__):
        net.load_state_dict(torch.load(__PATH__))
        # net.eval()  # 将模型转为评估模式
    modelTrain(train_loader, optimizer, net, criterion, __PATH__)
    modelTest(test_loader, net)
    torch.save(net.state_dict(), __PATH__)
