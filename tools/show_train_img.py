"""
@Project ：.ProjectCode 
@File    ：show_train_img
@Describe：
@Author  ：KlNon
@Date    ：2023/4/24 12:55 
"""
# obtain one batch of training images
import numpy as np
from matplotlib import pyplot as plt

from pytorch.model.args import dataloaders
from pytorch.model.label.model_load_label import cat_label_to_name
from tools.img_view import imgview


def show_train_img():
    dataiter = iter(dataloaders['train_data'])
    images, labels = dataiter.next()

    images = images.numpy()  # convert images to numpy for display

    # show some test images
    fig = plt.figure(figsize=(15, 15))
    fig_rows, fig_cols = 4, 5
    for index in np.arange(fig_rows * fig_cols):
        img = images[index]

        label = labels[index].item()
        title = f'Label: {label}\n{cat_label_to_name[label].title()}'

        ax = fig.add_subplot(fig_rows, fig_cols, index + 1, xticks=[], yticks=[])

        imgview(img, title, ax)
