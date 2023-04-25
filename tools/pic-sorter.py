"""
@Project ：.ProjectCode 
@File    ：pic-sorter
@Describe：当时自己做数据集的时候遗留下来的
@Author  ：KlNon
@Date    ：2023/3/13 16:55 
"""
import os
import shutil

# 按照类别将图像文件分组
classes = ['俄州黄金', '坎特公主', '奶油龙沙宝石', '彩蝶', '摩纳哥公爵', '无限永远', '果汁阳台', '梅郎口红', '泡芙美人', '浪漫宝贝', '绯扇', '绿野', '翰钱', '自由精神', '茶花女', '莱茵黄金', '蓝丝带', '蓝月', '金凤凰', '香云']  # 图像类别列表

# 图像数据集目录,当前目录回退一层再进入子目录
data_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "assets\\images")

for cls in classes:
    os.makedirs(os.path.join(data_dir, cls), exist_ok=True)

# 将PNG和JPG格式的图像文件复制到对应的子文件夹中
for filename in os.listdir(data_dir):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        src_file = os.path.join(data_dir, filename)
        for cls in classes:
            if cls in filename:
                dst_file = os.path.join(data_dir, cls, filename)
                shutil.move(src_file, dst_file)
                break
