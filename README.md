# flower-learning

该项目是对于蔷薇科植物生长的学习，使用了深度学习中的卷积神经网络 (CNN)，通过在训练集上训练模型，然后在测试集上评估模型在分类任务中的准确性。本项目使用 PyTorch 实现。

## 项目介绍

```
本项目使用了102种经典的英国常见花朵的图片数据集,同时添加了作者自己收集到的月季花数据集并进行了图像分割.
项目目前训练两个模型,一个是花朵种类识别,另一个是病虫害识别
```


## 使用方法

### 安装依赖
```shell
pip install -r requirements.txt
```
除此以外还需安装pytorch,请去官网 [PyTorch官方](https://pytorch.org/get-started/locally/) 安装

### 运行项目

1. 首先，在 assets/train 和 assets/test 目录下准备好合适的图片数据集。
2. 在 main.py文件中设置合适的参数,预处理代码已经在 'Net' 模型中设置, 可自行调整.
3. 运行以下命令开始训练和测试模型。

```shell
#训练模型
python main.py

#进行模型可视化测试
python use_model.py
```

## 代码结构

### 数据预处理

图像的预处理代码在 Net 模型中设置，包括以下过程：

- 随机旋转图像(旋转度数为40)
- 水平翻转图像
- 调整图像大小为 224x224
- 将图像转换为 Tensor 格式
- 对图像数据进行归一化

### 模型设计

在 model_net.py 中，定义了卷积神经网络模型 Net 。该模型有两个卷积层和两个全连接层，并在全连接层之间加入了一个 Dropout 层以防止过拟合。

### 模型训练

在 main.py 文件中，定义了训练模型的函数 modelTrain()，其中用到了 PyTorch 中的 DataLoader 和优化器，以及上述定义的损失函数、模型和停止训练的条件。

### 模型测试

在 main.py 文件中，定义了模型测试的函数 modelTest()，其中用到了上述定义的测试集 DataLoader。

在使用flask框架下的web进行测试时,请使用如下url进行测试
http://127.0.0.1:5000/inference?url=img
## 参考资料

- [PyTorch官方文档](https://pytorch.org/docs/stable/index.html)
- [pytorch-handbook](https://github.com/zergtant/pytorch-handbook)