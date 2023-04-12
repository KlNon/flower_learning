"""
@Project ：.ProjectCode 
@File    ：test_model
@Describe：
@Author  ：KlNon
@Date    ：2023/3/30 12:07 
"""

import torchvision
import numpy as np

from pytorch.model.init import *
from pytorch.model.net.model_net import Net

classes = ('丰花月季', '地被月季', '壮花月季', '大花香水月季', '微型月季', '树状月季', '灌木月季', '藤本月季')

dataiter = iter(test_loader)
images, labels = next(dataiter)

# print images
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

net = Net()
net.load_state_dict(torch.load(PATH))

outputs = net(images)

probability = nn.functional.softmax(outputs, dim=1)  # 计算softmax，即该图片属于各类的概率
max_value, index = torch.max(probability, 1)  # 找到最大概率对应的索引号，该图片即为该索引号对应的类别

print()
print("识别为'{}'的概率为{}".format(classes[index.argmax()], max_value.max().item()))

_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))
