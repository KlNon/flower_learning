import os

from pytorch.model.init import *
from pytorch.model.init import __PATH__
from pytorch.model.train.model_train import modelTrain
from pytorch.model.test.model_test import modelTest

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    if os.path.exists(__PATH__):
        net.load_state_dict(torch.load(__PATH__))
        # net.eval()  # 将模型转为评估模式
    modelTrain(train_loader, optimizer, net, criterion, __PATH__)
    modelTest(test_loader, net)
    torch.save(net.state_dict(), __PATH__)
