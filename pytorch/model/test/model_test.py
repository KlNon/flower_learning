"""
@Project ：.ProjectCode 
@File    ：model_test
@Describe：
@Author  ：KlNon
@Date    ：2023/3/29 22:07 
"""
import torch


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
