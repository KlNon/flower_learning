"""
@Project ：.ProjectCode 
@File    ：__init__
@Describe：
@Author  ：KlNon
@Date    ：2023/4/11 21:38 
"""

import torch.nn as nn

from pytorch.model.data_config import *
from pytorch.model.label.model_load_label import class_to_idx, cat_label_to_name

model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, momentum=0.9)


def freeze_parameters(root, freeze=True):
    [param.requires_grad_(not freeze) for param in root.parameters()]


# Save the checkpoint
def save_checkpoint(checkpoint_path='checkpoint.pt'):
    model.to('cpu')
    checkpoint = {'model_name': model_name,
                  'output_size': output_size,
                  'hidden_layers': hidden_layers,
                  'model_state_dict': model.state_dict(),
                  'class_to_idx': class_to_idx,
                  'cat_label_to_name': cat_label_to_name}

    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path='checkpoint.pt'):
    checkpoint = torch.load(checkpoint_path)

    model_name = checkpoint['model_name']
    output_size = checkpoint['output_size']
    hidden_layers = checkpoint['hidden_layers']

    model = create_network(model_name=model_name,
                           output_size=output_size, hidden_layers=hidden_layers)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.cat_label_to_name = checkpoint['cat_label_to_name']

    return model
