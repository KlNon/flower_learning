"""
@Project ：.ProjectCode 
@File    ：CV2
@Describe：
@Author  ：KlNon
@Date    ：2023/5/3 12:55 
"""
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pytorch.model.model_init import initialize_model, load_checkpoint

normalize_mean = np.array([0.485, 0.456, 0.406])
normalize_std = np.array([0.229, 0.224, 0.225])


def Main():
    checkpoint_dir, data_dir, device, image_datasets, dataloaders = initialize_model(which_file='Kind',
                                                                                     which_model='checkpoint',
                                                                                     output_size=103,
                                                                                     return_params=['checkpoint_dir',
                                                                                                    'data_dir',
                                                                                                    'device',
                                                                                                    'image_datasets',
                                                                                                    'dataloaders'])
    # 加载预训练的 Faster R-CNN 模型
    model = load_checkpoint(checkpoint_dir + 'checkpoint.pt')

    model.to(device)

    # 加载图像
    category = 30
    image_name = 'image_08077.jpg'
    image_path = data_dir + f'/valid/{category}/{image_name}'
    image = Image.open(image_path).convert('RGB')

    # 定义图像变换
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(
            normalize_mean,
            normalize_std)
    ])

    # 应用变换并添加批量维度
    input_image = transform(image).unsqueeze(0)

    input_image = input_image.to(device)
    # 使用模型进行预测
    with torch.no_grad():
        predictions = model(input_image)

    # 提取预测边界框、标签和分数
    boxes = predictions[0]['boxes']
    labels = predictions[0]['labels']
    scores = predictions[0]['scores']

    # 设置分数阈值
    threshold = 0.5

    # 绘制原始图像
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # 在图像上绘制边界框
    for box, label, score in zip(boxes, labels, scores):
        if score > threshold:
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1

            rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

    plt.show()


if __name__ == '__main__':
    Main()
