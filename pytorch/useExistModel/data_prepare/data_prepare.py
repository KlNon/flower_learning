"""
@Project ：.ProjectCode 
@File    ：data_prepare
@Describe：Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
@Author  ：KlNon
@Date    ：2023/4/25 14:24 
"""
import numpy as np
import torchvision.transforms.functional as TF

normalize_mean = np.array([0.485, 0.456, 0.406])
normalize_std = np.array([0.229, 0.224, 0.225])


def process_image(image, resize_size=256, crop_size=224, mean=normalize_mean, std=normalize_std):
    # Process a PIL image for use in a PyTorch model
    image = TF.resize(image, resize_size)

    upper_pixel = (image.height - crop_size) // 2
    left_pixel = (image.width - crop_size) // 2
    image = TF.crop(image, upper_pixel, left_pixel, crop_size, crop_size)

    image = TF.to_tensor(image)
    image = TF.normalize(image, mean, std)

    return image
