"""
@Project ：.ProjectCode 
@File    ：img_show
@Describe：
@Author  ：KlNon
@Date    ：2023/4/25 14:28 
"""
import numpy as np
from matplotlib import pyplot as plt

import numpy as np
import matplotlib.pyplot as plt


def imshow(image, ax=None, title=None, titlecolor='k'):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.grid(False)
    if title:
        ax.set_title(title, color=titlecolor)

    plt.show(block=True)
    return ax
