import torch
import torch.nn as nn
from torchvision import models

#需要看看论文是如何做的，目前这个不行
class Model1(nn.Module):

    def __init__(self, n_class):
        super().__init__()

        self.base_model = models.vgg16(pretrained=True)
        # self.base_model.children()
        #
        layers = list(self.base_model.children())
        self.layer1 = layers[0]
        # self.layer1 = nn.Sequential(*layers[:5])  # size=(N, 64, x.H/2, x.W/2)
        # self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')
        # self.layer2 = layers[5]  # size=(N, 128, x.H/4, x.W/4)
        # self.upsample2 = nn.Upsample(scale_factor=8, mode='bilinear')
        # self.layer3 = layers[6]  # size=(N, 256, x.H/8, x.W/8)
        # self.upsample3 = nn.Upsample(scale_factor=16, mode='bilinear')
        # self.layer4 = layers[7]  # size=(N, 512, x.H/16, x.W/16)
        # self.upsample4 = nn.Upsample(scale_factor=32, mode='bilinear')
        #
        # self.conv1k = nn.Conv2d(64 + 128 + 256 + 512, n_class, 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x=torch.flatten(x,1)
        # print('layer1:',x.size())
        # up1 = self.upsample1(x)
        # x = self.layer2(x)
        # up2 = self.upsample2(x)
        # x = self.layer3(x)
        # up3 = self.upsample3(x)
        # x = self.layer4(x)
        # up4 = self.upsample4(x)
        #
        # merge = torch.cat([up1, up2, up3, up4], dim=1)
        # merge = self.conv1k(merge)
        # out = self.sigmoid(merge)

        return x
