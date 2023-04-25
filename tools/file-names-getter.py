"""
@Project ：.ProjectCode 
@File    ：ile-names-getter
@Describe：获取文件名字
@Author  ：KlNon
@Date    ：2023/3/13 16:57 
"""
import os

# 获取当前工作目录
current_directory = os.getcwd()

# 获取当前目录的上一层目录
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))

# 进入子目录
subdirectory_path = os.path.join(parent_directory, "assets\\images")

# 进入目标目录
os.chdir(subdirectory_path)

# 打印当前工作目录
print(os.getcwd())

# 获取当前目录下的所有文件名
files = [os.path.splitext(filename)[0] for filename in os.listdir(subdirectory_path)]

# 打印文件名列表
print(files)
