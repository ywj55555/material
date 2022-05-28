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

# 得根据实际文件保存形式修改
class Dataset_all(torch.utils.data.Dataset):
    # file_list为文件列表
    def __init__(self, file_list,data_path):
        self.Data = file_list
        self.data_path = data_path
    def __getitem__(self, index):
        # img = npy_loader(self.data_path + '/envi/',self.Data[index]) #返回9通道数据
        # img = np.load(self.data_path + '/envi/'+ self.Data[index]+'.npy')
        # img = transform2(img) #特征设计
        img = np.load(self.data_path+'envi/'+ self.Data[index]+'.npy')
        label = np.load(self.data_path+'label/'+ self.Data[index]+'_label.npy')
        # label = transformlabel(label) # 把gt转成 0，1，2，3，4
        return img, label
    def __len__(self):
        return len(self.Data)

class MaterialSubModel(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels_1=16, mid_channels_2=32, mid_channels_3=8):
        super(MaterialSubModel, self).__init__()
        self.layer1 = nn.Sequential(
            # size - 2
            nn.Conv2d(in_channels, mid_channels_1, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU()
            #nn.ReLU()
            #nn.ReLU(inplace=True)
        )
        # size - 3
        self.layer2 = nn.Sequential(
            nn.Conv2d(mid_channels_1, mid_channels_2, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU()
            # nn.ReLU()
        )
        # size - 3
        self.layer3 = nn.Sequential(
            nn.Conv2d(mid_channels_2, mid_channels_3, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU()
            # nn.ReLU()
        )
        # size - 2
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

class MaterialBigModel(nn.Module):#11 * 11
    def __init__(self, in_channels = 21, class_num=2, mode='train' ,len_features = 50,mid_channel1 = 32):
        super(MaterialBigModel, self).__init__()
        feature = [
            # size - 2
            nn.Conv2d(in_channels, mid_channel1, stride=1, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(mid_channel1),
            # size - 3
            # nn.Conv2d(50, 100, stride=1, kernel_size=2, padding=0),
            nn.Conv2d(mid_channel1, mid_channel1 * 2, stride=1, kernel_size=4, padding=0),
            nn.Dropout(p=0.1),
            nn.ReLU(inplace=True),
            # 缩减一层
            # nn.Conv2d(mid_channel1 * 2, mid_channel1 * 2, stride=1, kernel_size=3, padding=1),
            # nn.Dropout(p=0.1),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm2d(mid_channel1 * 2),
            nn.Conv2d(mid_channel1 * 2, mid_channel1 * 3, stride=1, kernel_size=3, padding=1),
            nn.Dropout(p=0.2),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2),
            # size - 1
            nn.MaxPool2d(2, stride=1),
            # 缩减一层
            # nn.Conv2d(mid_channel1 * 3, mid_channel1 * 3, stride=1, kernel_size=3, padding=1),
            # nn.Dropout(p=0.2),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm2d(mid_channel1 * 3),
            nn.Conv2d(mid_channel1 * 3, mid_channel1 * 4, stride=1, kernel_size=3, padding=1),
            nn.Dropout(p=0.3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(mid_channel1 * 4),
            nn.Conv2d(mid_channel1 * 4, mid_channel1 * 5, stride=1, kernel_size=3, padding=1),
            nn.Dropout(p=0.3),
            nn.ReLU(inplace=True),
            # # 缩减一层
            # nn.Conv2d(mid_channel1 * 5, mid_channel1 * 5, stride=1, kernel_size=3, padding=1),
            # nn.Dropout(p=0.3),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm2d(mid_channel1 * 5),
            # nn.MaxPool2d(2, stride=2),
            # size - 1
            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(mid_channel1 * 5, mid_channel1 * 6, stride=1, kernel_size=3, padding=1),
            nn.Dropout(p=0.4),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(mid_channel1 * 6),
            nn.Conv2d(mid_channel1 * 6, mid_channel1 * 8, stride=1, kernel_size=3, padding=1),
            nn.Dropout(p=0.4),
            nn.ReLU(inplace=True),
            # nn.Conv2d(400, 400, stride=1, kernel_size=3, padding=1),
            # 缩减一层
            # nn.Conv2d(mid_channel1 * 8, mid_channel1 * 8, stride=1, kernel_size=3, padding=1),
            # nn.Dropout(p=0.4),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm2d(mid_channel1 * 8),
            # nn.MaxPool2d(2, stride=2),
            # size - 1
            nn.MaxPool2d(2, stride=1),
            # nn.Conv2d(400, 400, stride=1, kernel_size=1, padding=0)
            # size - 2
            nn.Conv2d(mid_channel1 * 8, mid_channel1 * 8, stride=1, kernel_size=3, padding=0)
        ]
        # 生成 1*1 进入分类器
        classifier = [
            nn.Conv2d(mid_channel1 * 8, mid_channel1 * 4, stride=1, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            # nn.Linear(900, 200),
            # nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel1 * 4, len_features, stride=1, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(len_features, class_num, stride=1, kernel_size=1, padding=0),
        ]
        self.mode = mode
        self.feature = nn.Sequential(*feature)
        self.classifier = nn.Sequential(*classifier)


    def forward(self, x):  # x: batch, window, slice channel, h, w
        input_tensor = x
        feat =  self.single_forward(input_tensor)
        return feat

    def single_forward(self, x):
        feat = self.feature(x)
        out = self.classifier(feat)
        return out

if __name__ == "__main__":
    # 1415, 1859
    dummy_input = torch.rand(1, 128, 1415, 1859).cuda()

    # print(dummy_input)
    model = MaterialBigModel(128,2,len_features = 32,mid_channel1 = 16).cuda().eval()
    for i in range(5):
        with torch.no_grad():
            predict = model(dummy_input)
            print(predict.shape)
