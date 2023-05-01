"""
@Project ：.ProjectCode 
@File    ：model_classifier
@Describe：创建网络之中的分类器(classifier)
@Author  ：KlNon
@Date    ：2023/4/24 12:45 
"""
from torch import nn
from collections import OrderedDict


def create_classifier(input_size, output_size, hidden_layers=None, dropout=0.5,
                      activation=nn.RReLU(), output_function=nn.LogSoftmax(dim=1)):
    if hidden_layers is None:
        hidden_layers = []

    layers = OrderedDict()

    # Add input layer
    if not hidden_layers:
        layers['layer0'] = nn.Linear(input_size, output_size)
    else:
        layers['layer0'] = nn.Linear(input_size, hidden_layers[0])
        layers['activ0'] = activation
        layers['drop_0'] = nn.Dropout(dropout)

        # Add hidden layers
        for i, (layer_in, layer_out) in enumerate(zip(hidden_layers[:-1], hidden_layers[1:])):
            layers[f'layer{i + 1}'] = nn.Linear(layer_in, layer_out)
            layers[f'activ{i + 1}'] = activation
            layers[f'drop_{i + 1}'] = nn.Dropout(dropout)

        # Add output layer
        layers['output'] = nn.Linear(hidden_layers[-1], output_size)

    if output_function:
        layers['output_function'] = output_function

    return nn.Sequential(layers)
