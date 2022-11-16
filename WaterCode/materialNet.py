import torch.nn as nn
import torch
from data.utilNetNew import *

class MaterialSubModel(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels_1=16, mid_channels_2=32, mid_channels_3=8):
        super(MaterialSubModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels_1, kernel_size=3, stride=1, padding=0),
            # nn.LeakyReLU()
            nn.ReLU()
            #nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(mid_channels_1, mid_channels_2, kernel_size=4, stride=1, padding=0),
            # nn.LeakyReLU()
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(mid_channels_2, mid_channels_3, kernel_size=4, stride=1, padding=0),
            # nn.LeakyReLU()
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(mid_channels_3, out_channels, kernel_size=3, stride=1, padding=0),
            #网络的最后一层最好不用relu激活，一般分类问题用softmax激活
            # nn.LeakyReLU()
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

class CNN(nn.Module):#16*16
    def __init__(self, in_channels = 21, class_num=2, mode='train' ,len_features = 50):
        super(CNN, self).__init__()
        feature = [
            nn.Conv2d(in_channels, 50, stride=1, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(50),
            nn.Conv2d(50, 100, stride=1, kernel_size=2, padding=0),
            # nn.Dropout(p=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(100, 100, stride=1, kernel_size=3, padding=1),
            # nn.Dropout(p=0.1),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(100),
            nn.Conv2d(100, 200, stride=1, kernel_size=3, padding=1),
            # nn.Dropout(p=0.2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(200, 200, stride=1, kernel_size=3, padding=1),
            # nn.Dropout(p=0.2),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(200),
            nn.Conv2d(200, 250, stride=1, kernel_size=3, padding=1),
            # nn.Dropout(p=0.3),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(250),
            nn.Conv2d(250, 300, stride=1, kernel_size=3, padding=1),
            # nn.Dropout(p=0.3),
            nn.ReLU(inplace=True),
            nn.Conv2d(300, 300, stride=1, kernel_size=3, padding=1),
            # nn.Dropout(p=0.3),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(300),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(300, 350, stride=1, kernel_size=3, padding=1),
            # nn.Dropout(p=0.4),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(350),
            nn.Conv2d(350, 400, stride=1, kernel_size=3, padding=1),
            # nn.Dropout(p=0.4),
            nn.ReLU(inplace=True),
            nn.Conv2d(400, 400, stride=1, kernel_size=3, padding=1),
            # nn.Dropout(p=0.4),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(400),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(400, 400, stride=1, kernel_size=1, padding=0)
        ]

        # select_module = [
        #     nn.Conv2d(in_channels, 256, stride=1, kernel_size=1, padding=0),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(256),
        #     nn.Conv2d(256, 512, stride=1, kernel_size=1, padding=0),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 512, stride=1, kernel_size=1, padding=0),
        #     nn.ReLU(inplace=True)
        #     ]

        # feature = [
        #     nn.Conv2d(11, 100, stride=1, kernel_size=3, padding=0),
        #     # nn.Dropout(p=0.2),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(100, 200, stride=1, kernel_size=2, padding=0),
        #     # nn.Dropout(p=0.2),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(200),
        #     nn.MaxPool2d(2, stride=2),

        #     nn.Conv2d(200, 250, stride=1, kernel_size=3, padding=1),
        #     # nn.Dropout(p=0.3),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(250),
        #     # nn.Conv2d(250, 300, stride=1, kernel_size=3, padding=1),
        #     nn.Dropout(p=0.3),
        #     nn.ReLU(inplace=True),
        #     # nn.Conv2d(300, 300, stride=1, kernel_size=3, padding=1),
        #     nn.Dropout(p=0.3),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(300),
        #     nn.MaxPool2d(2, stride=2),

        #     nn.Conv2d(300, 350, stride=1, kernel_size=3, padding=1),
        #     # nn.Dropout(p=0.4),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(350),
        #     nn.Conv2d(350, 400, stride=1, kernel_size=3, padding=1),
        #     # nn.Dropout(p=0.4),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(400, 400, stride=1, kernel_size=3, padding=1),
        #     # nn.Dropout(p=0.4),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(400),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.Conv2d(400, 400, stride=1, kernel_size=1, padding=0)
        # ]
        classifier = [
            nn.Conv2d(400, 200, stride=1, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),
            nn.Conv2d(200, len_features, stride=1, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(len_features, class_num, stride=1, kernel_size=1, padding=0),
        ]
        self.mode = mode
        # self.select_module = nn.Sequential(*select_module)
        self.feature = nn.Sequential(*feature)
        self.classifier = nn.Sequential(*classifier)
        # self.compare = nn.Parameter(torch.rand(4,11))


    def forward(self, x):  # x: batch, window, slice channel, h, w
        input_tensor = x
        feat =  self.single_forward(input_tensor)
        return feat

    def single_forward(self, x):
        # x = self.select_module(x)
        feat = self.feature(x)
        out = self.classifier(feat)
        return out

if __name__ == "__main__":
    dummy_input = torch.rand(10, 20, 11, 11)
    print(dummy_input)
    model = MaterialModel()
    predict = model(dummy_input)
    print(predict)
