"""
@Project ：.ProjectCode 
@File    ：test_model
@Describe：
@Author  ：KlNon
@Date    ：2023/3/30 12:07 
"""

import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

from pytorch.datasets.flower_dataset import *
from pytorch.model.net.model_net import Net
from pytorch.model.train.model_train import modelTrain
from pytorch.model.test.model_test import modelTest


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


classes = ('丰花月季', '地被月季', '壮花月季', '大花香水月季', '微型月季', '树状月季', '灌木月季', '藤本月季')

train_dataset = datasets.ImageFolder('./pytorch/datasets/train', transform=train_transform)
test_dataset = datasets.ImageFolder('./pytorch/datasets/test', transform=test_transform)
train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False, drop_last=True)

dataiter = iter(test_loader)
images, labels = next(dataiter)

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(20)))

net = Net()
PATH = 'pytorch/model/outdated/flower_net.pth'
net.load_state_dict(torch.load(PATH))

outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(20)))
