import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import DataLoader
from data.utilNetNew import *
from utils.load_spectral import kindsOfFeatureTransformation
from utils.load_spectral import envi_loader
from skimage import io


# band = [x - 1 for x in band]
bandNums = len(band)


waterLabelPath_test = '/home/glk/datasets/shenzhen_tiff/label/'#hefei_test2#
waterImgRootPath_test = '/home/glk/datasets/shenzhen_tiff/all_data/'#hefei_test


waterLabelPath_train = '/home/glk/datasets/shenzhen_tiff/label/'
waterImgRootPath_train = '/home/glk/datasets/shenzhen_tiff/all_data/'


def envi_normalize(imgData):
    # img_max =np.max(imgData, axis=0 ,keepdims = True)
    img_max =np.max(imgData,keepdims = True)
    # img_max = 1000
    return imgData / (img_max+0.0001)#65535#

class MyDataset_whole(torch.utils.data.Dataset):
    def __init__(self, train_set, dim = 128,feature_extraction = False, dataType = 'train'):
        self.dim = dim
        self.feature_extraction = feature_extraction
        if dataType == "train":
            data_file = "train" + train_set + ".txt"
            self.imgpath = waterImgRootPath_train
            data_file = self.imgpath + data_file
            self.label_path = waterLabelPath_train

        else:
            data_file = "test" + train_set + ".txt"  # Water_shenzhen
            self.imgpath = waterImgRootPath_test
            self.label_path = waterLabelPath_test
            data_file = self.imgpath + data_file


        with open(data_file,'r') as f:
            dataFile = f.readlines()
        self.img_list = dataFile

    def __getitem__(self, index):
        # Image = envi_loader(self.imgpath, self.img_list[index][3:].split('\n')[0], False)
        doc_dir = self.imgpath + self.img_list[index].split('\n')[0].split('.')[0]+'.tiff'
        Image = io.imread(doc_dir).astype(float).transpose(2,0,1)
        label = io.imread(self.label_path + self.img_list[index].split('\n')[0]).astype(int)
        Image = envi_normalize(Image[:-1,:,:])
        return Image, label

    def __len__(self):
        return len(self.img_list)