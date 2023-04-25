"""
@Project ：.ProjectCode 
@File    ：to_jpeg
@Describe：转换图片格式为jpg
@Author  ：KlNon
@Date    ：2023/3/29 17:21 
"""
import cv2
from PIL import Image
import os


# 将文件夹中的图像文件格式转换为JPEG格式
def convert_to_jpg(root):
    for subdir, dirs, files in os.walk(root):
        for file in files:
            filepath = os.path.join(subdir, file)
            ext = os.path.splitext(filepath)[1].lower()
            if ext not in ['.jpg', '.jpeg', '.png']:
                try:
                    img = Image.open(filepath)
                    new_filepath = os.path.splitext(filepath)[0] + '.jpg'
                    img.save(new_filepath, 'JPEG')
                    os.remove(filepath)
                except Exception as e:
                    print(f"Error converting {filepath}: {e}")


if __name__ == '__main__':
    convert_to_jpg(r'E:\.GraduationProject\ProjectCode\assets\train')
