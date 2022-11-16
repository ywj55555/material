from materialNet import *
from Dataloader import *
from utils.parse_args import parse_args
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch import nn
from utils.load_spectral import envi_loader
import io
import torch
import random
import cv2


args = parse_args()
# batchsize 也可以改 目前为32
mBatchSize = 1
train_set = 'data_all'
mEpochs = 100
mLearningRate = args.lr
dim = args.band_number
num_workers = args.num_workers
# 梯度更新步长 0.0001
mDevice=torch.device("cuda")
nora = args.nora
print('mBatchSize',mBatchSize)
print('mEpochs',mEpochs)
print('mLearningRate',mLearningRate)
print('nora',nora)
model_size = ["small", "big"]
featureTrans = args.featureTrans
class_nums = 3

class MyDataset_whole(torch.utils.data.Dataset):
    def __init__(self):
        train_set == 'data_all'
        waterLabelPath_train = '/home/glk/datasets/dataColorRgb/'
        waterImgRootPath_train = '/home/glk/datasets/dataColor/'


        data_file = train_set + ".txt"
        self.imgpath = waterImgRootPath_train
        data_file = self.imgpath + data_file
        self.label_path = waterLabelPath_train


        with open(data_file,'r') as f:
            dataFile = f.readlines()
        self.img_list = dataFile
        band = [6, 32, 54, 61, 77, 82, 87, 91, 95, 104, 108]
        band = [x + 5 for x in band]
        self.band = np.array(band)
        self.all_img = []
        for imgname in self.img_list:
            Image = envi_loader(self.imgpath, imgname[3:].split('\n')[0], True).transpose(2, 0, 1)
            tmp = Image[:, :, :].mean(0, keepdims=True)
            TMP = Image[self.band, :, :]
            for i in range(11):
                TMP[i] = Image[self.band[i] - 5:self.band[i] + 5, :, :].mean(0)
            Image = np.concatenate((TMP, tmp), axis=0)
            self.all_img.append(Image)


    def __getitem__(self, index):
        Image = self.all_img[index]
        label = cv2.imread(self.label_path + self.img_list[index].split('\n')[0] + '.tif')/255
        return Image, label

    def __len__(self):
        return len(self.img_list)

class CONV(nn.Module):
    """ Channel attention module"""
    def __init__(self):
        super(CONV, self).__init__()
        self.conv = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=1, stride=1)
    def forward(self,X):
        out = self.conv(X)
        return out

if __name__ == '__main__':
    seed = 2021
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    # 可以考虑改变benchmark为true
    torch.backends.cudnn.benchmark = False
    # 配合随机数种子，确保网络多次训练参数一致
    torch.backends.cudnn.deterministic = True
    # 使用非确定性算法
    torch.backends.cudnn.enabled = True

    trainDataset = MyDataset_whole()

    trainLoader = DataLoader(dataset=trainDataset, batch_size=mBatchSize, shuffle=True, num_workers = num_workers)#, pin_memory=False
    #
    class_nums = 3
    model = CONV().cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    scaler = GradScaler()
    model.train()
    for epoch in range(mEpochs):
        # 训练
        # model.train()
        trainCorrect = 0
        trainCorrect_spec = 0
        trainCorrect_spat = 0
        trainTotal = 0
        trainLossTotal = 0.0
        for i, data in enumerate(tqdm(trainLoader)):

            img, label = data
            img, label = Variable(img).float().cuda(), Variable(label).long().cuda()
            label = label.permute(0,3,1,2)
            # print(img.sum())
            predict = model(img)
            # print(predict.shape)
            # print(label.shape)
            loss = (predict - label).pow(2).mean()
            # print(loss)
            trainLossTotal += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            RGB = predict.squeeze().permute(1,2,0).cpu().detach().numpy()*255
            # print(RGB)
            # print(RGB.shape)
            cv2.imwrite('./save_rgb/' + str(i) + '.png',RGB)
        print('epoch:',epoch, 'loss:',trainLossTotal)

