"""
@Project ：.ProjectCode 
@File    ：use_model
@Describe：
@Author  ：KlNon
@Date    ：2023/4/24 22:42
"""
import torch

from pytorch.model.args import gdrive_dir, device
from pytorch.model.init import load_checkpoint
from pytorch.model.test.model_test import modelTest

if __name__ == '__main__':
    # model = load_checkpoint(gdrive_dir + 'checkpoint.pt')
    #
    # model.to(device)
    #
    # modelTest(model, show_graphs=True)
    print(torch.version)
