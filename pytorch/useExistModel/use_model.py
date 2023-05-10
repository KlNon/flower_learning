"""
@Project ：.ProjectCode 
@File    ：use_model
@Describe：使用matplotlib库可视化测试模型
@Author  ：KlNon
@Date    ：2023/4/24 22:42
"""
from matplotlib import pyplot as plt

from pytorch.model.label.model_load_label import load_labels
from pytorch.model.model_init import *
from pytorch.model.test.model_test import modelTest
from pytorch.useExistModel.img.img_show import imshow


def main(num_of_classes=5):
    model = load_checkpoint(checkpoint_dir + 'checkpoint.pt')  # 加载预先训练好的模型
    model.to(device)  # 将模型移至相应设备（CPU或GPU）

    model.eval()  # 将模型设置为评估模式（关闭dropout和batch normalization等）
    with torch.no_grad():  # 在计算过程中禁用梯度计算，以节省内存和加速计算
        for images, labels in dataloaders['test_data']:  # 对测试数据集中的每个批次进行循环

            # Move tensors to device
            images, labels = images.to(device), labels.to(device)  # 将图像和标签移至相应设备（CPU或GPU）

            # Get predictions for this test batch
            output = model(images)  # 使用模型对图像进行预测

            output = torch.exp(output)  # 将输出转换为概率
            probs, classes = output.topk(num_of_classes, dim=1)  # 获取每个图像的前5个最高概率和对应的类别

            # Display an image along with the top 5 classes
            images = images.cpu()  # 将图像移回CPU
            probs = probs.data.cpu()  # 将概率移回CPU
            classes = classes.data.cpu()  # 将类别移回CPU

            # 创建一个新的matplotlib图形，用于显示图像和对应的概率
            fig = plt.figure(figsize=(30, 56),
                             tight_layout={'pad': 0, 'w_pad': 5, 'h_pad': 0, 'rect': (0, 0, 1, 1)})

            rows = 4  # 每行显示2个子图
            lines = 30  # 显示32行子图（总共64个子图）
            line_ctrl = 0  # 用于控制行数的变量

            for index in range(len(images)):  # 对批次中的每个图像进行循环
                if index % rows == 0:  # 检查是否需要增加行数
                    line_ctrl += 1

                prob = probs[index].squeeze()  # 获取当前图像的概率

                clazz = [cat_label_to_name[c.item()].title() for c in classes[index]]  # 获取当前图像的类别名称

                label = labels[index].item()  # 获取当前图像的真实标签
                title = f'{cat_label_to_name[label].title()}'  # 获取真实标签对应的类别名称

                position = index + 1 + (line_ctrl * rows)  # 计算子图的位置
                ax1 = fig.add_subplot(lines, rows, position, xticks=[], yticks=[])  # 在图形上添加一个子图用于显示图像

                titlecolor = 'g'  # 如果预测正确，标题显示为绿色
                if title != clazz[0]:  # 如果预测错误，标题显示为红色
                    titlecolor = 'r'

                ax2 = fig.add_subplot(lines, rows, position + rows, xticks=[], yticks=[])  # 在图形上添加一个子图用于显示概率
                bars = ax2.barh(np.arange(num_of_classes), prob)  # 创建水平条形图
                ax2.set_yticks(np.arange(num_of_classes))
                ax2.set_yticklabels(clazz)
                ax2.set_ylim(-1, 5)
                ax2.invert_yaxis()
                ax2.set_xlim(0, 1.1)
                imshow(images[index], ax1, title, title_color=titlecolor)

                # 在条形图上添加百分比概率作为标签
                for bar, p in zip(bars, prob):
                    width = bar.get_width()
                    percentage = p * 100  # 将概率值乘以100得到百分比
                    ax2.text(width + 0.01, bar.get_y() + bar.get_height() / 2, f'{percentage:.1f}%', va='center')

            plt.show()
            break


if __name__ == '__main__':
    checkpoint_dir, device, image_datasets, dataloaders = initialize_model(which_file='Kind',
                                                                           which_model='checkpoint',
                                                                           output_size=103,
                                                                           return_params=['checkpoint_dir', 'device',
                                                                                          'image_datasets',
                                                                                          'dataloaders'])

    model = load_checkpoint(checkpoint_dir + 'checkpoint.pt')  # 加载预先训练好的模型
    cat_label_to_name, class_to_idx = load_labels(image_datasets, file_name='kind_cat_to_name.json')
    modelTest(model.to(device), dataloader=dataloaders['test_data'], device=device, show_graphs=False, save_graphs=False)
    main(num_of_classes=5)

    checkpoint_dir, device, image_datasets, dataloaders = initialize_model(which_file='Diseases',
                                                                           which_model='checkpoint1',
                                                                           output_size=8,
                                                                           return_params=['checkpoint_dir', 'device',
                                                                                          'image_datasets',
                                                                                          'dataloaders'])

    model = load_checkpoint(checkpoint_dir + 'checkpoint.pt')  # 加载预先训练好的模型
    cat_label_to_name, class_to_idx = load_labels(image_datasets, file_name='diseases_cat_to_name.json')
    modelTest(model.to(device), dataloader=dataloaders['test_data'], device=device, show_graphs=False, save_graphs=False)
    main(num_of_classes=5)

    checkpoint_dir, device, image_datasets, dataloaders = initialize_model(which_file='Water',
                                                                           which_model='checkpoint2',
                                                                           output_size=4,
                                                                           return_params=['checkpoint_dir', 'device',
                                                                                          'image_datasets',
                                                                                          'dataloaders'])

    model = load_checkpoint(checkpoint_dir + 'checkpoint.pt')  # 加载预先训练好的模型
    cat_label_to_name, class_to_idx = load_labels(image_datasets, file_name='water_cat_to_name.json')
    modelTest(model.to(device), dataloader=dataloaders['test_data'], device=device, show_graphs=False, save_graphs=False)
    main(num_of_classes=4)

