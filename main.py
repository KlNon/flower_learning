import os

from pytorch.model.init import *
from pytorch.model.train.model_train import modelTrain
from pytorch.model.val.model_val import modelVal
from pytorch.model.test.model_test import modelTest

# 按间距中的绿色按钮以运行脚本。

if __name__ == '__main__':
    if os.path.exists(PATH):
        net.load_state_dict(torch.load(PATH))
    modelTrain(optimizer, net, criterion)
    net.eval()
    modelTest(net)
    torch.save(net.state_dict(), PATH)
