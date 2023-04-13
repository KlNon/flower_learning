"""
@Project ：.ProjectCode
@File    ：flask
@Describe：
@Author  ：KlNon
@Date    ：2023/4/13 16:41
"""
from flask import Flask, request
from pytorch.model.net.model_net import Net

app = Flask(__name__)

'''
卷积神经网络相关程序
'''
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models  # 人家的模型
from torch.autograd import Variable
import torch
from torch import nn

# 数据预处理
data_transform = transforms.Compose([
    transforms.RandomRotation(40),  # 随机旋转度数
    transforms.RandomHorizontalFlip(),  # 水平翻转
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),  # 将图像转换为Tensor
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.229, 0.224, 0.225)),  # 数据归一化
])

# 类别
data_classes = ('丰花月季', '地被月季', '壮花月季', '大花香水月季', '微型月季', '树状月季', '灌木月季', '藤本月季')

# 选择CPU还是GPU的操作
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 选择模型
net = Net()
net.load_state_dict(torch.load('../flower_net.pth'))

# net.load_state_dict(torch.load("VGG16_flower_200.pkl", map_location=torch.device('cpu')))
net.eval()
net.to(device)

'''
flask相关程序
'''


@app.route('/inference')
def inference():
    im_url = request.args.get('url')

    # 读取数据
    img = Image.open(im_url)
    img = data_transform(img)  # 这里经过转换后输出的input格式是[C,H,W],网络输入还需要增加一维批量大小B
    img = img.unsqueeze(0)  # 增加一维，输出的img格式为[1,C,H,W]
    img = img.to(device)

    with torch.no_grad():
        score = net(img)
        probability = nn.functional.softmax(score, dim=1)
        max_value, index = torch.max(probability.cpu(), 1)

    return str(data_classes[index.item()])


if __name__ == '__main__':
    app.run()
