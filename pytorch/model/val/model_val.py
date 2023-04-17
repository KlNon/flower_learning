"""
@Project ：.ProjectCode 
@File    ：model_val
@Describe：
@Author  ：KlNon
@Date    ：2023/4/16 22:22 
"""
import torch

from pytorch.model.args import *
from pytorch.model.init import criterion

data_classes = data_classes


def modelVal(net):
    # 使用验证集评估模型性能
    with torch.no_grad():
        val_loss = 0.0
        correct = 0
        total = 0
        for data in val_loader:
            if device.type == 'cuda':
                inputs, targets = data[0].to(device), data[1].to(device)
            else:
                inputs, targets = data

            outputs = net(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        val_accuracy = 100 * correct / total
        val_loss /= len(val_loader)

    # 输出验证损失、准确率等指标
    print('Validation Loss: {:.4f}, Validation Accuracy: {:.2f} %'
          .format(val_loss, val_accuracy))
