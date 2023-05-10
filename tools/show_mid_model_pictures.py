"""
@Project ：.ProjectCode 
@File    ：show_mid_model_pictures
@Describe：展示每一层过程中的特征图
@Author  ：KlNon
@Date    ：2023/5/9 3:14 
"""
import numpy as np
import torch
from torchvision.transforms import transforms
from PIL import Image
import matplotlib.pyplot as plt

from pytorch.model.net.model_net import create_network

normalize_mean = np.array([0.485, 0.456, 0.406])
normalize_std = np.array([0.229, 0.224, 0.225])

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载预训练的ResNet50模型
model = load_checkpoint('./../checkpoint/checkpoint.pt')  # 加载预先训练好的模型
model = model.to(device)
model.eval()

# 加载并预处理图像
image_path = "./../assets/Kind/test/0/image_06736.jpg"
image_transform = transforms.Compose([
        transforms.RandomChoice([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(180),
        ]),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            normalize_mean,
            normalize_std)
])

image = Image.open(image_path)
image_tensor = image_transform(image).unsqueeze(0)
image_tensor = image_tensor.to(device)

# 定义钩子函数
activations = []


def hook(module, input, output):
    activations.append(output)


# 为每个卷积层添加钩子
for name, layer in model.named_modules():
    if isinstance(layer, torch.nn.modules.conv.Conv2d):
        layer.register_forward_hook(hook)

# 通过模型传递图像
output = model(image_tensor)

# 绘制特征图
for i, activation in enumerate(activations):
    num_features = activation.shape[1]
    rows = int(num_features ** 0.5)
    cols = int(num_features / rows) + 1

    fig = plt.figure(figsize=(cols * 2, rows * 2))
    fig.suptitle(f"Feature maps at layer {i + 1}")

    for j in range(num_features):
        feature_map = activation[0, j].detach().cpu().numpy()
        ax = fig.add_subplot(rows, cols, j + 1)
        ax.imshow(feature_map, cmap="gray")
        ax.axis("off")

    plt.savefig('./mid_pictures/' + str(i) + '.png')
    # plt.show()
