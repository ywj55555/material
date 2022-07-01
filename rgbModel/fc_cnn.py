import torch
import torch.nn as nn
from torchvision import models

class FcCNN(nn.Module):
    def __init__(self, n_class, train=True):
        super(FcCNN,self).__init__()
        self.base_model = models.vgg16(pretrained=train)
        layers = list(self.base_model.children())
        self.layer1 = layers[0]

        self.linear = nn.Linear(512, n_class)

    def forward(self, input):
        #输入是 32*32*3的rgb图像
        x = self.layer1(input)
        x = x.view(x.size(0), -1)
        #print('size:',x.size())
        x=self.linear(x)
        return x