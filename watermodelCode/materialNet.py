import torch.nn as nn
from torch.utils.data import DataLoader
from data.utilNetNew import *


# 问问高斯核怎么弄
# 数据加载模板
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


# class MaterialSubModel(nn.Module):
#     def __init__(self, in_channels, out_channels, mid_channels_1=16, mid_channels_2=32, mid_channels_3=8):
#         super(MaterialSubModel, self).__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels_1, kernel_size=3, stride=1, padding=0),
#             # 使用relu的话 网络会失火 全部爆0
#             nn.LeakyReLU()
#             #nn.ReLU()
#             #nn.ReLU(inplace=True)
#         )
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(mid_channels_1, mid_channels_2, kernel_size=4, stride=1, padding=0),
#             nn.LeakyReLU()
#             # nn.ReLU()
#         )
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(mid_channels_2, mid_channels_3, kernel_size=4, stride=1, padding=0),
#             nn.LeakyReLU()
#             # nn.ReLU()
#         )
#         self.layer4 = nn.Sequential(
#             nn.Conv2d(mid_channels_3, out_channels, kernel_size=3, stride=1, padding=0),
#             #网络的最后一层最好不用relu激活，一般分类问题用softmax激活
#             # nn.LeakyReLU()
#             # nn.ReLU()
#         )
#
#     def forward(self, x):
#         x=self.layer1(x)
#         #print('layer1:',x.size())
#         x=self.layer2(x)
#         #print('layer2:', x.size())
#         x = self.layer3(x)
#         #print('layer3:', x.size())
#         x = self.layer4(x)
#         #x = x.view(x.size(0),-1)
# #        print('layer4:', x.size())
#         #x=self.linear(x)
#         return x

# 也可以改成双流网络
class MaterialSubModel(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels_1=16, mid_channels_2=32, mid_channels_3=8):
        super(MaterialSubModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels_1, kernel_size=3, stride=1, padding=0),
            # 使用relu的话 网络会失火 全部爆0
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

class WaterModel(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels_1=110, mid_channels_2=220, mid_channels_3=110, finalChannel = 64):
        super(WaterModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels_1, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(mid_channels_1),
            # 使用relu的话 网络会失火 全部爆0
            # nn.LeakyReLU()
            nn.ReLU()
            #nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(mid_channels_1, mid_channels_2, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(mid_channels_2),
            # nn.LeakyReLU()
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(mid_channels_2, mid_channels_3, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(mid_channels_3),
            # nn.LeakyReLU()
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(mid_channels_3, finalChannel, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(finalChannel),
            nn.Conv2d(finalChannel, int(finalChannel * 0.5), kernel_size=1, stride=1, padding=0),
            nn.Conv2d(int(finalChannel * 0.5),out_channels, kernel_size=1, stride=1, padding=0),
            #网络的最后一层最好不用relu激活，一般分类问题用softmax激活
            # nn.LeakyReLU()
            # nn.ReLU()
        )

    def forward(self, x):
        x=self.layer1(x)
        # print('layer1:',x)
        x=self.layer2(x)
        #print('layer2:', x.size())
        x = self.layer3(x)
        #print('layer3:', x.size())
        x = self.layer4(x)
        #x = x.view(x.size(0),-1)
#        print('layer4:', x.size())
        #x=self.linear(x)
        return x



if __name__ == "__main__":
    dummy_input = torch.rand(10, 20, 11, 11)
    print(dummy_input)
    # model = MaterialModel()
    # predict = model(dummy_input)
    print(predict)
