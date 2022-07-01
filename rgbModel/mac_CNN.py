import torch
import torch.nn as nn
from torchvision import models
#使用交叉熵
class MAC_CNN(nn.Module):
    def __init__(self, n_class):
        super(MAC_CNN,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2,2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2)
        )

        #self.linear1 = nn.Linear(4608, n_class)
        self.linear1 = nn.Linear(512, n_class)

    def forward(self, input):
        # 输入是 16*16*3的rgb图像
        #print('input:',input.size())
        x = self.layer1(input)
        #print('size1:',x.size())
        x = self.layer2(x)
        #print('size2:',x.size())
        x = self.layer3(x)
        #print('size3:',x.size())
        x = self.layer4(x)
        #print('size5:', x.size())
        x = x.view(x.size(0), -1)
        #print('size5:',x.size())
        x=self.linear1(x)
        #print('size2:', x.size())
        #x = self.linear2(x)
        return x