"""
@Project ：.ProjectCode 
@File    ：model_picture
@Describe：查看网络结构
@Author  ：KlNon
@Date    ：2023/5/7 20:37 
"""
import torch
from torchsummary import summary

from pytorch.model.net.model_net import create_network


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


if __name__ == '__main__':
    model = load_checkpoint('./../checkpoint/checkpoint.pt')  # 加载预先训练好的模型
    model = model.to('cuda')
    summary(model, (3, 224, 224))
