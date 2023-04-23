import torch.nn as nn


class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        self.features = model.features
        # for p in self.parameters():
        #     p.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, 103, bias=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x
