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

def evaluate(model, criterion, dataloader, device):
    loss = 0
    accuracy = 0
    top_class_graph = []
    labels_graph = []

    with torch.no_grad():
        for images, labels in dataloader:
            labels_graph.extend(labels)
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            batch_loss = criterion(output, labels)
            loss += batch_loss.item() * len(images)
            output = torch.exp(output)
            top_ps, top_class = output.topk(1, dim=1)
            top_class_graph.extend(top_class.view(-1).to('cpu').numpy())
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.sum(equals.type(torch.FloatTensor)).item()

    return loss / len(dataloader.dataset), accuracy / len(dataloader.dataset), top_class_graph, labels_graph


def save_graph(filename, labels_graph, top_class_graph):
    plt.figure(figsize=(25, 13))
    plt.plot(np.array(labels_graph), 'k.')
    plt.plot(np.array(top_class_graph), 'r.')
    plt.savefig(filename)
    plt.close()


def modelTest(model, dataloader, device, show_graphs=True, save_graphs=True):
    model.eval()
    criterion = nn.NLLLoss()
    test_loss, accuracy, top_class_graph, labels_graph = evaluate(model, criterion, dataloader, device)

    log = f'Test Loss: {test_loss:.6f}\tTest accuracy: {(accuracy * 100):.2f}%'
    print(log)

    if show_graphs or save_graphs:
        for i, (images, labels) in enumerate(dataloader):
            if show_graphs:
                save_graph(f'output_data/test/TestImg_batch_{i + 1}.png', labels_graph, top_class_graph)
            if save_graphs:
                save_graph(f'output_data/testTestImg.png', labels_graph, top_class_graph)
