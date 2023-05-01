"""
@Project ：.ProjectCode
@File    ：model_classifier
@Describe：创建网络
@Author  ：KlNon
@Date    ：2023/4/24 12:45
"""

from torchvision import models

# Build and train your network
from torchvision.models import VGG19_Weights, ResNet50_Weights, ResNet152_Weights

from pytorch.model.net.classifier.model_classifier import create_classifier


def create_network(model_name='resnet50', output_size=103, hidden_layers=[1000]):
    if model_name == 'resnet50':
        # Download the model
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # Replace the model classifier
        model.fc = create_classifier(2048, output_size, hidden_layers)

        return model

    if model_name == 'resnet152':
        # Download the model
        model = models.resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
        # Replace the model classifier
        model.fc = create_classifier(2048, output_size, hidden_layers)

        return model

    if model_name == 'vgg19':
        # Download the model
        model = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        # Replace the model classifier
        model.fc = create_classifier(2048, output_size, hidden_layers)

        return model

    return None
