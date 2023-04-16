"""
@Project ：.ProjectCode 
@File    ：data_divide
@Describe：划分数据集(训练集,验证集,测试集)
@Author  ：KlNon
@Date    ：2023/4/15 13:04 
"""
import os
import random
# import shutil
from shutil import copy2

# 比例
scale = [0.6, 0.2, 0.2]

# 类别
classes = ['丰花月季', '切花月季', '大花月季', '微型月季', '藤本月季', '其它月季']

for each in classes:
    datadir_normal = "./../assets/data/" + each + "/"  # 原文件夹

    all_data = os.listdir(datadir_normal)  # （图片文件夹）
    num_all_data = len(all_data)
    print(each + "类图片数量: " + str(num_all_data))
    index_list = list(range(num_all_data))
    # print(index_list)
    random.shuffle(index_list)
    num = 0

    trainDir = "./../assets/train/" + each  # （将训练集放在这个文件夹下）
    if not os.path.exists(trainDir):  # 如果不存在这个文件夹，就创造一个
        os.makedirs(trainDir)

    validDir = "./../assets/val/" + each  # （将验证集放在这个文件夹下）
    if not os.path.exists(validDir):
        os.makedirs(validDir)

    testDir = "./../assets/test/" + each  # （将测试集放在这个文件夹下）
    if not os.path.exists(testDir):
        os.makedirs(testDir)

    for i in index_list:
        fileName = os.path.join(datadir_normal, all_data[i])
        if num < num_all_data * scale[0]:
            # print(str(fileName))
            copy2(fileName, trainDir)
        elif num_all_data * scale[0] < num < num_all_data * (scale[0] + scale[1]):
            # print(str(fileName))
            copy2(fileName, validDir)
        else:
            copy2(fileName, testDir)
        num += 1
