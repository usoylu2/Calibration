import torch.nn as nn
import torch
import torchvision.models as models


class MNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64,  kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.premodel = nn.Sequential(*list(models.resnet50(pretrained=True).children())[1:-1])
        self.fc = nn.Linear(2048, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.premodel(x)
        x = torch.flatten(x, 1)# flatten all dimensions except batch
        x = self.fc(x)
        return x


class MNetdense(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64,  kernel_size=(7, 7), stride=(2, 2), padding=(5, 5), bias=False)
        self.premodel = nn.Sequential(*list(list(models.densenet121(pretrained=True).children())[0].children())[1:])
        self.fc = nn.Linear(6144, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.premodel(x)
        x = torch.flatten(x, 1)# flatten all dimensions except batch
        x = self.fc(x)
        return x


class MNet_bce(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64,  kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.premodel = nn.Sequential(*list(models.resnet50(pretrained=True).children())[1:-1])
        self.fc = nn.Linear(2048, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.premodel(x)
        x = torch.flatten(x, 1)# flatten all dimensions except batch
        x = self.fc(x)
        return x


class MNetdense_bce(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64,  kernel_size=(7, 7), stride=(2, 2), padding=(5, 5), bias=False)
        self.premodel = nn.Sequential(*list(list(models.densenet201(pretrained=True).children())[0].children())[1:])
        self.fc = nn.Linear(11520, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.premodel(x)
        x = torch.flatten(x, 1)# flatten all dimensions except batch
        x = self.fc(x)
        return x
