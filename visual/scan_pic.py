"""
@Project ：.ProjectCode 
@File    ：scan_pic
@Describe：
@Author  ：KlNon
@Date    ：2023/5/10 0:24 
"""
import torch
import torchvision.transforms as transforms

from pytorch.model.net.model_net import create_network

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_dir = '../checkpoint/'
checkpoint1_dir = '../checkpoint1/'
checkpoint2_dir = '../checkpoint2/'


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


def load_model():
    model1 = load_checkpoint(checkpoint_dir + 'checkpoint.pt').to(DEVICE)  # 加载预先训练好的模型
    model1.eval()
    model2 = load_checkpoint(checkpoint1_dir + 'checkpoint.pt').to(DEVICE)  # 加载预先训练好的模型
    model2.eval()
    model3 = load_checkpoint(checkpoint2_dir + 'checkpoint.pt').to(DEVICE)  # 加载预先训练好的模型
    model3.eval()
    return model1, model2, model3


def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 根据训练模型时所用的归一化参数进行修改
    ])

    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch


def get_prediction(img, model):
    input_batch = preprocess_image(img)
    input_batch = input_batch.to(DEVICE)
    with torch.no_grad():
        logits = model(input_batch)

    probs, classes = logits.topk(1, dim=1)  # 获取每个图像的前5个最高概率和对应的类别
    return probs, classes
