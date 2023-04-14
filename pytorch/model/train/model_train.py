"""
@Project ：.ProjectCode 
@File    ：model_train
@Describe：在训练过程中也进行测试
@Author  ：KlNon
@Date    ：2023/3/29 22:03
"""

# 训练模型
import torch

from pytorch.model.init import device
from pytorch.model.test.model_test import modelTest


def modelTrain(train_loader, test_loader, optimizer, net, criterion, path, max_epochs=200, min_loss=0.01):
    for epoch in range(1, max_epochs + 1):
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

        avg_loss = running_loss / len(train_loader)
        print('[%d/%5d] loss: %.3f' % (epoch, len(train_loader), avg_loss))

        if epoch + 1 % (len(train_loader) // 8):
            print('Saving model after %d epochs' % epoch)
            torch.save(net.state_dict(), path)
            modelTest(test_loader, net)

        if avg_loss < min_loss or epoch >= len(train_loader) * 2:
            print('Training finished after %d epochs' % epoch)
            torch.save(net.state_dict(), path)
            return

    print('Training finished after %d epochs' % max_epochs)
    torch.save(net.state_dict(), path)
