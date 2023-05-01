"""
@Project ：.ProjectCode 
@File    ：model_load_label
@Describe：读取json文件,将其序号与具体内容对应
@Author  ：KlNon
@Date    ：2023/4/24 12:49 
"""
import json


def load_labels(image_datasets, file_name='cat_to_name.json'):
    with open('E:/.GraduationProject/ProjectCode/pytorch/model/label/' + file_name, 'r') as f:
        cat_to_name = json.load(f)

    class_to_idx = image_datasets['train_data'].class_to_idx

    cat_label_to_name = {}
    for cat, label in class_to_idx.items():
        name = cat_to_name.get(cat)
        cat_label_to_name[label] = name

    print(cat_label_to_name)
    return cat_label_to_name, class_to_idx
