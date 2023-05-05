"""
@Project ：.ProjectCode 
@File    ：model_train
@Describe：使用已经训练好的resnet50模型来训练fasterRCNN模型,以便可以显示边界框
@Author  ：KlNon
@Date    ：2023/5/3 13:32 
"""

# 使用RCNN模型进行边界检测
import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import resnet50

from main import load_model
from pytorch.model.model_init import checkpoint_base_, model_file

# 加载预训练的 ResNet50 模型


checkpoint_path = checkpoint_base_ + model_file + '/' + 'checkpoint_phase_three.pt'
checkpoint = torch.load(checkpoint_path)
resnet_model = resnet50(pretrained=False)

fc_optimizer = torch.optim.Adagrad(checkpoint.fc.parameters(), lr=0.01, weight_decay=0.001)
resnet_model.load_state_dict(load_model(checkpoint_path, checkpoint, fc_optimizer))

# 删除 ResNet50 的全连接层，以便将其用作骨干网络
backbone = torch.nn.Sequential(*list(resnet_model.children())[:-2])

# 定义 Faster R-CNN 的锚点生成器
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))

# 使用预训练的 ResNet50 骨干网络创建 Faster R-CNN 模型
model = FasterRCNN(backbone,
                   num_classes=91,  # 根据您的数据集设置类别数
                   rpn_anchor_generator=anchor_generator)

# 在这里，您可以使用您的数据集对 Faster R-CNN 模型进行训练，或将其用于物体检测和边界框回归。
torch.save(model.state_dict(), './../RCNNmodel/detection_model.pt')
