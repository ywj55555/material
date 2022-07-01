import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
from utilNet import *
import time
import random
import pytorch_colors as colors

mBatchSize=32
mEpochs=300
mLearningRate=0.0001
mDevice=torch.device("cuda")
class MyDataset(torch.utils.data.Dataset):
    def __init__(self,inputData,inputLabel):
        self.Data=inputData
        self.Label=inputLabel
    def __getitem__(self, index):
        img=self.Data[index]
        label=self.Label[index]
        return img,label
    def __len__(self):
        return len(self.Data)
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
            nn.Conv2d(in_channels=8, out_channels=3, kernel_size=3, stride=1,padding=0),
            nn.ReLU()
        )

    def forward(self,input):
        x=self.layer1(input)
        x=self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

def yCbCr2rgb(input_im):
    im_flat = input_im.contiguous().view(-1, 3).float()
    mat = torch.tensor([[1.164, 1.164, 1.164],
                       [0, -0.392, 2.017],
                       [1.596, -0.813, 0]]).to(mDevice)
    bias = torch.tensor([-16.0/255.0, -128.0/255.0, -128.0/255.0]).to(mDevice)
    temp = (im_flat + bias).mm(mat)
    out = temp.view(3, list(input_im.size())[1], list(input_im.size())[2])
    return out

if __name__=='__main__':
    seed = 2018
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

    trainData,trainLabel=generateData('train',5)
    testData,testLabel=generateData('test',3)


    trainDataset=MyDataset(trainData,trainLabel)
    testDataset=MyDataset(testData,testLabel)
    trainLoader=DataLoader(dataset=trainDataset,batch_size=mBatchSize,shuffle=True)
    testLoader=DataLoader(dataset=testDataset,batch_size=mBatchSize,shuffle=True)


    model=MaterialModel().cuda()
    criterion=nn.MSELoss()
    #criterion=nn.CrossEntropyLoss()
    #optimizer=torch.optim.SGD(model.parameters(),lr=mLearningRate) #或 Adam
    optimizer=torch.optim.Adam(model.parameters(),lr=mLearningRate)

    for epoch in range(mEpochs):
        # 训练
        trainCorrect=0
        trainTotal=0
        for i,data in enumerate(trainLoader,0):
            img,label=data
            img, label = Variable(img).float().cuda(), Variable(label).float().cuda()
            tmp = torch.Tensor(img.size(0), 7, img.size(2), img.size(3)).cuda()
            print(img[:, :3].size())
            print(type(img[:, :3]))

            tmp[:, :3]=colors.rgb_to_yuv(img[:, :3])

            tmp[:, 4] = img[:, 0] - img[:, 1]
            tmp[:, 5] = img[:, 1] - img[:, 2]
            tmp[:, 6] = img[:, 2] - img[:, 3]
            tmp[:, 3] = img[:, 3]
            #label=torch.unsqueeze()
            trainTotal += label.size(0)
 
            predict = model(tmp)
            predict = torch.squeeze(predict)
            #计算正确率
            predictIndex = torch.argmax(predict, dim=1)
            labelIndex = torch.argmax(label, dim=1)
            trainCorrect += (predictIndex == labelIndex).sum()
            #产生loss
            #loss = criterion(predict, label.item())
            loss = criterion(predict, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('train epoch:',epoch,': ',trainCorrect.item()/trainTotal)
        torch.save(model.state_dict(), "model/test/shinei/params" + str(epoch) + ".pkl")
        # 测试
        testCorrect = 0
        testTotal = 0
        for i, data in enumerate(testLoader, 0):
            img, label = data
            img, label = Variable(img).float().cuda(), Variable(label).float().cuda()
            tmp = torch.Tensor(img.size(0), 7, img.size(2), img.size(3)).to(mDevice)
            tmp[:, :3] = img[:, :3]
            tmp[:, 4] = img[:, 0] - img[:, 1]
            tmp[:, 5] = img[:, 1] - img[:, 2]
            tmp[:, 6] = img[:, 2] - img[:, 3]
            tmp[:, 3] = img[:, 3]
            testTotal += label.size(0)
            predict = model(tmp)
            predict = torch.squeeze(predict)
            # 计算正确率
            predictIndex = torch.argmax(predict, dim=1)
            labelIndex = torch.argmax(label, dim=1)
            testCorrect += (predictIndex == labelIndex).sum()
        print('test epoch:', epoch, ': ', testCorrect.item() /testTotal)
