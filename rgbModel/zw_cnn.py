# import torch
import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.autograd import Variable
# from torch.utils.data import DataLoader
# import numpy as np
# import time
# import random
# import pytorch_colors as colors
class MaterialModel(nn.Module):
    def __init__(self):
        super(MaterialModel,self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(in_channels=7, out_channels=16, kernel_size=3, stride=1,padding=0),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=1,padding=0),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=4, stride=1,padding=0),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1,padding=0),
            nn.ReLU()
        )

    def forward(self,input):
        x=self.layer1(input)
        x=self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class MaterialModel_leakrelu(nn.Module):
    def __init__(self):
        super(MaterialModel_leakrelu,self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(in_channels=7, out_channels=16, kernel_size=3, stride=1,padding=0),
            # nn.ReLU()
            nn.LeakyReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=1,padding=0),
            # nn.ReLU()
            nn.LeakyReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=4, stride=1,padding=0),
            # nn.ReLU()
            nn.LeakyReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1,padding=0),
            # nn.ReLU()
            nn.LeakyReLU()
        )

    def forward(self,input):
        x=self.layer1(input)
        x=self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x