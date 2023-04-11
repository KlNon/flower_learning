"""
@Project ：.ProjectCode 
@File    ：model_train
@Describe：
@Author  ：KlNon
@Date    ：2023/3/29 22:03 
"""


# 训练模型
import torch


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
