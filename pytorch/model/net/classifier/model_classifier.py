"""
@Project ：.ProjectCode 
@File    ：model_classifier
@Describe：
@Author  ：KlNon
@Date    ：2023/4/24 12:45 
"""
from torch import nn
from collections import OrderedDict


def create_classifier(input_size, output_size, hidden_layers=None, dropout=0.5,
                      activation=nn.RReLU(), output_function=nn.LogSoftmax(dim=1)):
    if hidden_layers is None:
        hidden_layers = []
    dict = OrderedDict()

    if len(hidden_layers) == 0:
        dict['layer0'] = nn.Linear(input_size, output_size)

    else:

        dict['layer0'] = nn.Linear(input_size, hidden_layers[0])
        if activation:
            dict['activ0'] = activation
        if dropout:
            dict['drop_0'] = nn.Dropout(dropout)

        # for layer_in, layer_out in range(len(hidden_layers)):
        for layer, layer_in in enumerate(zip(hidden_layers[:-1], hidden_layers[1:])):
            dict['layer' + str(layer + 1)] = nn.Linear(layer_in[0], layer_in[1])
            if activation:
                dict['activ' + str(layer + 1)] = activation
            if dropout:
                dict['drop_' + str(layer + 1)] = nn.Dropout(dropout)

        dict['output'] = nn.Linear(hidden_layers[-1], output_size)

    if output_function:
        dict['output_function'] = output_function

    return nn.Sequential(dict)
