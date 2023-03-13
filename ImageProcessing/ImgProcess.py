"""
@Project ：.ProjectCode 
@File    ：ImgProcess
@Describe：
@Author  ：KlNon
@Date    ：2023/3/13 0:08 
"""

import cv2
import numpy as np

# 加载图像
img = cv2.imread('test.png')

# 调整大小
resized_img = cv2.resize(img, (224, 224))

# 裁剪
cropped_img = img[50:200, 50:200]

# 增强
flipped_img = cv2.flip(img, 1)
blurred_img = cv2.GaussianBlur(img, (5, 5), 0)

# 去除噪声
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)[1]

# 标准化
normalized_img = img / 255.0

# 数据增强
M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), 30, 1.0)
rotated_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

# 保存图像
cv2.imwrite('resized_img.jpg', resized_img)
cv2.imwrite('cropped_img.jpg', cropped_img)
cv2.imwrite('flipped_img.jpg', flipped_img)
cv2.imwrite('blurred_img.jpg', blurred_img)
cv2.imwrite('thresh_img.jpg', thresh_img)
cv2.imwrite('normalized_img.jpg', normalized_img)
cv2.imwrite('rotated_img.jpg', rotated_img)
