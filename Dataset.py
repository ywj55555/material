import torch.nn as nn
from torch.utils.data import DataLoader
from data.utilNetNew import *
import os

env_data_dir = '/data3/chenjialin/hangzhou_data/envi/'
env_sh_data_dir = '/data3/chenjialin/hangzhou_data/envi_shanghai/'
label_data_dir = '/data3/chenjialin/hangzhou_data/label/'

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
    def __init__(self, file_list,data_path):
        self.Data = file_list
        self.data_path = data_path
    def __getitem__(self, index):
        # img = npy_loader(self.data_path + '/envi/',self.Data[index]) #返回9通道数据
        # img = np.load(self.data_path + '/envi/'+ self.Data[index]+'.npy')
        # img = transform2(img) #特征设计
        img = np.load(self.data_path+'envi/'+ self.Data[index]+'.npy')
        label = np.load(self.data_path+'label/'+ self.Data[index]+'_label.npy')
        label = transformlabel(label) # 把gt转成 0，1，2，3，4
        return img, label
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
