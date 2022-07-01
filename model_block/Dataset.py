import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from data.utilNetNew import *
import os
import sys
import cv2
# from utils.load_spectral import
png_path = '/home/cjl/ssd/dataset/shenzhen/rgb/needmark1/'
hz_png_path = '/home/cjl/ssd/dataset/hangzhou/rgb/'
hz_label = '/home/cjl/ssd/dataset/hangzhou/label/'
waterLabelPath = '/home/cjl/ssd/dataset/shenzhen/label/Label_rename/'
# waterImgRootPath = 'D:/ZF2121133HHX/water/daytime/'
waterImgRootPath = '/home/cjl/ssd/dataset/shenzhen/img/train/'
hangzhou_img_path = '/home/cjl/ssd/dataset/hangzhou/'

label_dict = {'__skin':1,'_cloth':2,'_plant':3,'_other':0}
label2target = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

class MyDataset_ori(torch.utils.data.Dataset):
    def __init__(self, inputData, inputLabel):
        self.Data = inputData
        self.Label = inputLabel

    def __getitem__(self, index):
        img = self.Data[index]
        label = self.Label[index]
        return img, label

    def __len__(self):
        return len(self.Data)

class Dataset_patch_mem(torch.utils.data.Dataset):
    def __init__(self, inputData, inputLabel):
        self.Data = inputData
        self.Label = inputLabel

    def __getitem__(self, index):
        img = self.Data[index]
        label = self.Label[index]
        return img, label

    def __len__(self):
        return len(self.Data)
class Dataset_patch(torch.utils.data.Dataset):
    # file_list为文件列表
    def __init__(self, file_list,data_path):
        self.Data = file_list
        self.data_path = data_path

    def __getitem__(self, index):
        img = np.load(self.data_path + self.Data[index])
        img = img[5:-5,5:-5]
        label = label2target[label_dict[self.Data[index][-14:-8]]]
        label = np.array(label)
        return img, label

    def __len__(self):
        return len(self.Data)

class Dataset_all(torch.utils.data.Dataset):
    # file_list为文件列表
    def __init__(self, file_list, img_path, label_path, select_train_bands, nora, featureTrans, activate=None):
        self.Data = file_list
        self.img_path = img_path
        self.label_path = label_path
        self.select_train_bands = select_train_bands
        self.activate = activate

        self.featureTrans = featureTrans
        self.nora = nora

    def __getitem__(self, index):
        imgLabel = io.imread(self.label_path + self.Data[index] + '.png')
        imgData = None
        if os.path.exists(self.img_path + self.Data[index][3:] + '.img'):
            imgData = envi_loader(self.img_path, self.Data[index][3:], self.select_train_bands, False)
        else:
            print(self.Data[index][3:] + '.img not exist!!!')
            sys.exit()
        # t3 = time.time()
        # 没必要 特征变换 增加之前设计的斜率特征
        if imgData is None:
            print("Not Found ", self.Data[index][3:])
            sys.exit()
        # H W 22
        if self.featureTrans:
            print("kindsOfFeatureTransformation......")
            # 11 -》 21
            imgData = kindsOfFeatureTransformation_slope(imgData, self.activate, self.nora)
        else:
            if self.nora:
                # print("normalizing......")
                imgData = envi_wholeMaxnormalize(imgData)
        return imgData, imgLabel
    def __len__(self):
        return len(self.Data)

class DatasetSpectralAndRgb(torch.utils.data.Dataset):
    # file_list为文件列表
    def __init__(self, file_list, img_path, label_path, rgb_path, select_train_bands, nora, featureTrans, activate=None):
        self.Data = file_list
        self.img_paths = img_path
        self.rgb_paths = rgb_path
        self.label_paths = label_path
        self.select_train_bands = select_train_bands
        self.activate = activate
        self.featureTrans = featureTrans
        self.nora = nora

    def __getitem__(self, index):
        if not isinstance(self.label_paths,list):
            self.label_paths = [self.label_paths]
        if not isinstance(self.rgb_paths,list):
            self.rgb_paths = [self.rgb_paths]
        if not isinstance(self.img_paths,list):
            self.img_paths = [self.img_paths]

        imgLabel = None
        for label_path in self.label_paths:
            if os.path.exists(label_path + self.Data[index] + '.png'):
                imgLabel = io.imread(label_path + self.Data[index] + '.png')
                break
        if imgLabel is None:
            print(self.Data[index] + '.png not exist!!!')
            sys.exit()

        imgData = None
        for img_path in self.img_paths:
            if os.path.exists(img_path + self.Data[index][3:] + '.img'):
                imgData = envi_loader(img_path, self.Data[index][3:], self.select_train_bands, False)
                if self.featureTrans:
                    print("kindsOfFeatureTransformation......")
                    # 11 -》 21
                    imgData = kindsOfFeatureTransformation_slope(imgData, self.activate, self.nora)
                else:
                    if self.nora:
                        imgData = envi_normalize(imgData)
                break
        if imgData is None:
            print("Not Found ", self.Data[index][3:])
            sys.exit()

        rgbData = None
        for rbg_path in self.rgb_paths:
            if os.path.exists(rbg_path + self.Data[index] + '.png'):
                rgbData = cv2.imread(rbg_path + self.Data[index] + '.png')  # 加载模式为 BGR
                rgbData = rgbData.astype(np.float64)[:, :, ::-1]  # 转为 RGB 进行训练
                break
        if rgbData is None:
            print(self.Data[index] + '.png RGB not exist!!!')
            sys.exit()
        return imgData, rgbData.copy(), imgLabel
        # return imgData_tmp, rgbData_tmp.copy(), imgLabel
    def __len__(self):
        return len(self.Data)

class Dataset_RGB(torch.utils.data.Dataset):
    # file_list为文件列表
    def __init__(self, file_list, png_path, label_path):
        self.Data = file_list
        self.png_path = png_path
        self.label_path = label_path
    def __getitem__(self, index):
        imgLabel = io.imread(self.label_path + self.Data[index] + '.png')
        imgData = cv2.imread(self.png_path + self.Data[index] + '.png')  # 加载模式为 BGR
        imgData = imgData.astype(np.float64)[:, :, ::-1]  # 转为 RGB 进行训练
        # 下面这些步骤记得在 训练的时候 实现
        # imgData = imgData / 255.0
        # imgData -= mean
        # imgData /= std
        #
        # image = image.transpose((2, 0, 1))
        # image = torch.from_numpy(image)
        #
        # #  image = image.permute((2, 0, 1))
        #
        # image = image.unsqueeze(0)
        return imgData.copy(), imgLabel
    def __len__(self):
        return len(self.Data)

def read_list(filepath):
    # if datatepy=='train':
    img_list = []
    label_list = []
    filelist = os.listdir(filepath)
    for file in filelist:
        img = np.load(filepath + file)
        img = img[5:-5, 5:-5]
        img_list.append(img)
        label = label2target[label_dict[file[-14:-8]]]
        label = np.array(label)
        label_list.append(label)
        # label = np.array(label)
    return img_list, label_list

def read_list_test(filepath):
    # if datatepy=='train':
    img_list = []
    label_list = []
    filelist = os.listdir(filepath)[::50000]
    for file in filelist:
        img = np.load(filepath + file)
        img = img[5:-5, 5:-5]
        img_list.append(img)
        label = label2target[label_dict[file[-14:-8]]]
        label = np.array(label)
        label_list.append(label)

    return img_list, label_list
