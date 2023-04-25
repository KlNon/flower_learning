"""
@Project ：.ProjectCode 
@File    ：img_view
@Describe：图片展示
@Author  ：KlNon
@Date    ：2023/4/24 12:53 
"""
import numpy as np

from pytorch.model.data_config import normalize_std, normalize_mean


def imgview(img, title, ax):
    # un-normalize
    for i in range(img.shape[0]):
        img[i] = img[i] * normalize_std[i] + normalize_mean[i]

    # convert from Tensor image
    ax.imshow(np.transpose(img, (1, 2, 0)))

    ax.set_title(title)
