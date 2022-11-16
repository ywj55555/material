import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import DataLoader
from data.utilNetNew import *
from utils.load_spectral import kindsOfFeatureTransformation
from utils.load_spectral import envi_loader

band = [2,36,54,61,77,82,87,91,95,104,108]
band = [x + 5 for x in  band]

# band = [x - 1 for x in band]
bandNums = len(band)

# waterLabelPath_test = '/home/glk/datasets/Water_shenzhen/label/'
# waterImgRootPath_test = '/home/glk/datasets/Water_shenzhen/'
waterLabelPath_test = '/home/glk/datasets/hefei/All_data/label/'#hefei_test2#
waterImgRootPath_test = '/home/glk/datasets/hefei/All_data/'#hefei_test
# waterLabelPath_test = '/home/glk/datasets/hangzhou/label/'
# waterImgRootPath_test = '/home/glk/datasets/hangzhou/'

# waterLabelPath_train = '/home/glk/datasets/Water_shenzhen/label/'
# waterImgRootPath_train = '/home/glk/datasets/Water_shenzhen/'
waterLabelPath_train = '/home/glk/datasets/water_data_mix/label/'
waterImgRootPath_train = '/home/glk/datasets/water_data_mix/'
# waterLabelPath_train = '/home/glk/datasets/hefei/All_data/label/'
# waterImgRootPath_train = '/home/glk/datasets/hefei/All_data/'

def envi_normalize(imgData):
    # img_max =np.max(imgData, axis=0 ,keepdims = True)
    img_max =np.max(imgData,keepdims = True)
    # img_max = 1000
    return imgData / (img_max+0.0001)#65535#

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, train_set, inputData, inputLabel, dim = 128,feature_extraction = False, dataType = 'train', size=11):
        self.Data = inputData
        self.Label = inputLabel
        self.dim = dim
        self.feature_extraction = feature_extraction
        if dataType == "train":
            data_file = "/home/glk/datasets/Water_shenzhen/train" + train_set + ".txt"
        else:
        # dataFile = waterFileTest
            data_file = "/home/glk/datasets/Water_shenzhen/test" + train_set + ".txt"#Water_shenzhen
        self.imgpath = waterImgRootPath
        with open(data_file,'r') as f:
            dataFile = f.readlines()
        self.img_list = dataFile
        self.size = size//2

    def __getitem__(self, index):
        Image = envi_loader(self.imgpath, self.img_list[index][3:].split('\n')[0], False).transpose(2,0,1)
        select_list = np.where(self.Data[:,2]==index)
        pixal = self.Data[select_list[0],:2]
        label = self.Label[select_list[0]]
        Image = envi_normalize(Image)
        if self.dim == 11:
            img = Image[band, :, :]

        else:
            img = Image
        input_data = []

        for i in range(pixal.shape[0]):
            x = pixal[i][0]
            y = pixal[i][1]
            img_patch = img[:,int(x-self.size):int(x+self.size+1),int(y-self.size):int(y+self.size+1)]
            if self.feature_extraction:
                img_patch = img_patch.transpose(1,2,0)
                img_patch = kindsOfFeatureTransformation(img_patch,nora=False).transpose(2,0,1)
            input_data.append(img_patch)
        img = np.array(input_data)
        return img, label

    def __len__(self):
        return len(self.img_list)
# class MyDataset(torch.utils.data.Dataset):
#     def __init__(self, inputData, inputLabel, dim = 128,feature_extraction = False, dataType = 'train', size=11):
#         self.Data = inputData
#         self.Label = inputLabel
#         self.dim = dim
#         self.feature_extraction = feature_extraction
#         if dataType == "train":
#             data_file = "/home/glk/datasets/Water_shenzhen/train.txt"
#         else:
#         # dataFile = waterFileTest
#             data_file = "/home/glk/datasets/Water_shenzhen/test.txt"
#         self.imgpath = waterImgRootPath
#         with open(data_file,'r') as f:
#             dataFile = f.readlines()
#         self.img_list = dataFile
#         self.prev_img = 0
#         self.Image = np.load(self.imgpath + self.img_list[0][3:].split('\n')[0] + '.npy')
#         #envi_loader(self.imgpath, self.img_list[0][3:].split('\n')[0], False)
#         self.size = size//2

#     def __getitem__(self, index):
#         img_num = int(self.Data[index][-1])
#         pixal = self.Data[index][:2]
#         if img_num == self.prev_img:
#             img_whole = self.Image
#         else:
#             print('new_image')
#             print(img_num)
#             img_whole = envi_loader(self.imgpath, self.img_list[img_num][3:].split('\n')[0], False)
#             self.Image = img_whole
#             self.prev_img = img_num
#         img = np.array(img_whole[int(pixal[0]-self.size):int(pixal[0]+self.size+1),int(pixal[1]-self.size):int(pixal[1]+self.size+1),:], dtype=np.float32).transpose(2,0,1)
#         if self.dim == 11:
#             img = img[band, :, :]
#         img = envi_normalize(img)
#         if self.feature_extraction:
#             img = img.transpose(1,2,0)
#             img = kindsOfFeatureTransformation(img,nora=False).transpose(2,0,1)
#         label = self.Label[index]
#         return img, label

#     def __len__(self):
#         return len(self.Data)

class MyDataset_whole(torch.utils.data.Dataset):
    def __init__(self, train_set, dim = 128,feature_extraction = False, dataType = 'train'):
        self.dim = dim
        self.feature_extraction = feature_extraction
        if train_set == '_a_HF':
            self.Hefei = True
            waterLabelPath_train = '/home/glk/datasets/hefei/All_data/label/'
            waterImgRootPath_train = '/home/glk/datasets/hefei/All_data/'
        elif train_set == '_all_HS':
            self.Hefei = True
            waterLabelPath_train = '/home/glk/datasets/water_data_mix/label/'
            waterImgRootPath_train = '/home/glk/datasets/water_data_mix/'
        elif train_set == '_x_F':
            self.Hefei = True
            waterLabelPath_train = '/home/glk/datasets/hefei/All_data/label/'
            waterImgRootPath_train = '/home/glk/datasets/hefei/All_data/'
        else:
            self.Hefei = False
            waterLabelPath_train = '/home/glk/datasets/water_data_mix/label/'
            waterImgRootPath_train = '/home/glk/datasets/water_data_mix/'
            # waterLabelPath_train = '/home/glk/datasets/Water_shenzhen/label/'
            # waterImgRootPath_train = '/home/glk/datasets/Water_shenzhen/'

        if dataType == "train":
            data_file = "train" + train_set + ".txt"
            self.imgpath = waterImgRootPath_train
            data_file = self.imgpath + data_file
            self.label_path = waterLabelPath_train

        else:
        # dataFile = waterFileTest
        #     data_file = "/home/glk/datasets/Water_shenzhen/test" + train_set + ".txt"
        #     data_file = "/home/glk/datasets/hefei/hefei_test/test" + train_set + ".txt"
            if train_set == '_a_HS':
                train_set = "_a"#
            else:
                train_set = "_all"#"_a"#"_x"#
            train_set = "_a_HF"  #'_all_data_without_label'#
            data_file = "test" + train_set + ".txt"  # Water_shenzhen
            self.imgpath = waterImgRootPath_test
            self.label_path = waterLabelPath_test
            data_file = self.imgpath + data_file


        with open(data_file,'r') as f:
            dataFile = f.readlines()
        self.img_list = dataFile
        self.band = np.array(band)
        if self.dim == 13:
            self.band = np.array([114, 109, 125,  53, 108,  81, 100, 112,  25,  90,  96, 123 ])

    def __getitem__(self, index):
        Image = envi_loader(self.imgpath, self.img_list[index][3:].split('\n')[0], False).transpose(2,0,1)
        label = io.imread(self.label_path + self.img_list[index].split('\n')[0] + '.png')
        Image = envi_normalize(Image)
        tmp = Image[:, :, :].mean(0,keepdims=True)
        if self.dim == 11 or self.dim == 12:
            Image = Image[self.band, :, :]

            # img += Image[self.band - 1, :, :] * 0.67
            # img += Image[self.band + 1, :, :] * 0.67
            # img += Image[self.band + 2, :, :] * 0.33
            # img += Image[self.band - 2, :, :] * 0.33
            # img = img / 3
            # print(img.shape)
            if self.dim == 12:
                Image = np.concatenate((Image,tmp),axis=0)
            # print(img.shape)
        elif self.dim == 13:
            Image = Image[self.band, :, :]
            Image = np.concatenate((Image, tmp), axis=0)
        else:
            Image = Image
        if self.Hefei:
            label = label * -1 + 2
        return Image, label

    def __len__(self):
        return len(self.img_list)