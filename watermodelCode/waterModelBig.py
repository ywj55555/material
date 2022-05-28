import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import DataLoader
from data.utilNetNew import *


# 问问高斯核怎么弄
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, inputData, inputLabel):
        self.Data = inputData
        self.Label = inputLabel

    def __getitem__(self, index):
        img = self.Data[index]
        label = self.Label[index]
        return img, label

    def __len__(self):
        return len(self.Data)


class MaterialSubModel(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels_1=16, mid_channels_2=32, mid_channels_3=8):
        super(MaterialSubModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels_1, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU()
            #nn.ReLU()
            #nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(mid_channels_1, mid_channels_2, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU()
            # nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(mid_channels_2, mid_channels_3, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU()
            # nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(mid_channels_3, out_channels, kernel_size=3, stride=1, padding=0),
            #网络的最后一层最好不用relu激活，一般分类问题用softmax激活
            nn.LeakyReLU()
            # nn.ReLU()
        )

    def forward(self, x):
        x=self.layer1(x)
        #print('layer1:',x.size())
        x=self.layer2(x)
        #print('layer2:', x.size())
        x = self.layer3(x)
        #print('layer3:', x.size())
        x = self.layer4(x)
        #x = x.view(x.size(0),-1)
#        print('layer4:', x.size())
        #x=self.linear(x)
        return x


class MaterialModel(nn.Module):
    def __init__(self):
        super(MaterialModel, self).__init__()
        self.subModel_skin = MaterialSubModel(8, 2)
        self.subModel_cloth = MaterialSubModel(6, 2)
        self.subModel_plant = MaterialSubModel(6, 2)
        self.linear = nn.Linear(6, 4)

    def forward(self, x):
        x1 = x[:, 0:8, :, :]
        x1 = self.subModel_skin(x1)
        #print(x1.size())
        x2 = x[:, 8:14, :, :]
        x2 = self.subModel_cloth(x2)
        #print(x2.size())
        x3 = x[:, 14:20, :, :]
        x3 = self.subModel_plant(x3)
        #print(x3.size())
        x = torch.cat([x1, x2, x3], 1)
        x = x.squeeze()
        x = x.squeeze()
        #print(x.size())
        x = self.linear(x)
        return x

class CNNCTC(nn.Module):
    def __init__(self, class_num=2, mode='train' ,len_features = 50):
        super(CNNCTC, self).__init__()
        feature = [
            nn.Conv2d(20, 50, stride=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(50),
            nn.Conv2d(50, 100, stride=1, kernel_size=3, padding=1),
            nn.Dropout(p=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(100, 100, stride=1, kernel_size=3, padding=1),
            nn.Dropout(p=0.1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(100),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(100, 200, stride=1, kernel_size=3, padding=1),
            nn.Dropout(p=0.2),
            nn.ReLU(inplace=True),
            nn.Conv2d(200, 200, stride=1, kernel_size=3, padding=1),
            nn.Dropout(p=0.2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(200),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(200, 250, stride=1, kernel_size=3, padding=1),
            nn.Dropout(p=0.3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(250),
            nn.Conv2d(250, 300, stride=1, kernel_size=3, padding=1),
            nn.Dropout(p=0.3),
            nn.ReLU(inplace=True),
            nn.Conv2d(300, 300, stride=1, kernel_size=3, padding=1),
            nn.Dropout(p=0.3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(300),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(300, 350, stride=1, kernel_size=3, padding=1),
            nn.Dropout(p=0.4),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(350),
            nn.Conv2d(350, 400, stride=1, kernel_size=3, padding=1),
            nn.Dropout(p=0.4),
            nn.ReLU(inplace=True),
            nn.Conv2d(400, 400, stride=1, kernel_size=3, padding=1),
            nn.Dropout(p=0.4),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(400),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(400, 400, stride=1, kernel_size=1, padding=0)
        ]

        classifier = [
            # nn.Linear(400, 200),
            nn.Conv2d(400, 200, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            # nn.Linear(900, 200),
            # nn.ReLU(inplace=True),
            # 全连接层恐怕不太行吧
            # nn.Linear(200, len_features)
            nn.Conv2d(200, len_features, kernel_size=1, stride=1, padding=0)
        ]
        self.mode = mode
        self.feature = nn.Sequential(*feature)
        self.classifier = nn.Sequential(*classifier)
        # self.fn = nn.Linear(len_features , class_num)
        self.fullConv = nn.Conv2d(len_features, class_num , kernel_size=1, stride=1, padding=0)

    def forward(self, x):  #
        input_tensor = x
        feat =  self.single_forward(input_tensor)
        return feat

    def single_forward(self, x):
        feat = self.feature(x)
        # feat = feat.view(feat.shape[0], -1)  # flatten
        out = self.classifier(feat)
        out = self.fullConv(out)
        return out

if __name__ == "__main__":
    dummy_input = torch.rand(10, 20, 16, 16)
    print(dummy_input)
    model = CNNCTC()
    predict = model(dummy_input)
    print(predict.shape)
