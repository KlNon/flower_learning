"""
@Project ：.ProjectCode 
@File    ：model_test
@Describe：
@Author  ：KlNon
@Date    ：2023/3/29 22:07 
"""
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

from pytorch.model.args import data_classes, dataloaders, device

classes = data_classes


def modelTest(model, dataloader=dataloaders['test_data'], show_graphs=True):
    #####################
    #       TEST        #
    #####################
    criterion = nn.NLLLoss()
    test_loss = 0
    accuracy = 0
    top_class_graph = []
    labels_graph = []
    # Set model to evaluation mode
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            labels_graph.extend(labels)

            # Move tensors to device
            images, labels = images.to(device), labels.to(device)

            # Get predictions for this test batch
            output = model(images)

            # Calculate loss for this test batch
            batch_loss = criterion(output, labels)
            # Track validation loss
            test_loss += batch_loss.item() * len(images)

            # Calculate accuracy
            output = torch.exp(output)
            top_ps, top_class = output.topk(1, dim=1)
            top_class_graph.extend(top_class.view(-1).to('cpu').numpy())
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.sum(equals.type(torch.FloatTensor)).item()

    #####################
    #     PRINT LOG     #
    #####################

    # calculate average losses
    test_loss = test_loss / len(dataloader.dataset)
    accuracy = accuracy / len(dataloader.dataset)

    # print training/validation statistics
    log = f'Test Loss: {test_loss:.6f}\t\
           Test accuracy: {(accuracy * 100):.2f}%'
    print(log)

    if show_graphs:
        plt.figure(figsize=(25, 13))
        plt.plot(np.array(labels_graph), 'k.')
        plt.plot(np.array(top_class_graph), 'r.')
        # plt.show()
        #
        # plt.show()
        plt.savefig('TestImg.png')
        plt.close()
