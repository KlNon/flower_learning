"""
@Project ：.ProjectCode 
@File    ：model_load_label
@Describe：
@Author  ：KlNon
@Date    ：2023/4/24 12:49 
"""
import json

from pytorch.model.args import image_datasets

with open('E:/.GraduationProject/ProjectCode/pytorch/model/label/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

class_to_idx = image_datasets['train_data'].class_to_idx

cat_label_to_name = {}
for cat, label in class_to_idx.items():
    name = cat_to_name.get(cat)
    cat_label_to_name[label] = name

# print(cat_label_to_name)
