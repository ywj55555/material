import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import DataLoader
from data.utilNetNew import *
from utils.load_spectral import kindsOfFeatureTransformation
import spectral.io.envi as envi



def judegHdrDataType(hdr_dirpath, file):
    with open(hdr_dirpath + file + '.hdr', "r") as f:
        data = f.readlines()
    modify_flag = False
    if data[5] != 'data type = 12\n':
        data[5] = 'data type = 12\n'
        modify_flag = True
        # raise HdrDataTypeError("data type = 2, but data type should be 12")
    if data[6].split(' =')[0] != 'byte order':
        data.insert(6,'byte order = 0\n')
        modify_flag = True
    else:
        if data[6] != 'byte order = 0\n':
            data[6] ='byte order = 0\n'
            modify_flag = True
    if modify_flag:
        with open(hdr_dirpath + file + '.hdr', "w") as f:
            f.writelines(data)
        print("mend the datatype of file : ", file)

def envi_loader(dirpath, filename,norma=True):
    judegHdrDataType(dirpath, filename)
    enviData = envi.open(dirpath + filename + '.hdr', dirpath + filename + '.img')
    imgData = enviData.load()
    imgData = np.array(imgData)
    if norma:
        imgData = envi_normalize(imgData)
    gc.collect()
    return imgData

def envi_normalize(imgData):
    # img_max =np.max(imgData, axis=2 ,keepdims = True)
    img_max =np.max(imgData,keepdims = True)
    return imgData / 65535#(img_max+0.0001)

bandNums = 128


def envi_normalize(imgData):
    # img_max =np.max(imgData, axis=0 ,keepdims = True)
    img_max =np.max(imgData,keepdims = True)
    # img_max = 1000
    return imgData / (img_max+0.0001)#65535#


class MyDataset_whole(torch.utils.data.Dataset):
    def __init__(self, train_set, dim = 128,feature_extraction = False, dataType = 'train'):
        self.dim = dim
        self.feature_extraction = feature_extraction
        LabelPath = '/home/glk/datasets/Multispec/label_4/'#label
        ImgRootPath = '/home/glk/datasets/Multispec/'
#train_set = ‘_x’
        if dataType == "train":
            data_file = "train" + train_set + ".txt"
            self.imgpath = ImgRootPath
            data_file = self.imgpath + data_file
            self.label_path = LabelPath

        else:
            data_file = "test" + train_set + ".txt"  # Water_shenzhen
            self.imgpath = ImgRootPath
            self.label_path = LabelPath
            data_file = self.imgpath + data_file


        with open(data_file,'r') as f:
            dataFile = f.readlines()
        self.img_list = dataFile

    def __getitem__(self, index):
        Image = envi_loader(self.imgpath, self.img_list[index].split('\n')[0], False).transpose(2,0,1)
        label = io.imread(self.label_path + self.img_list[index].split('/')[-1].split('\n')[0] + '.png')
        Image = envi_normalize(Image)
        return Image, label

    def __len__(self):
        return len(self.img_list)


class DataPrefetcher:
    """
    DataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.
    """

    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.input_cuda = self._input_cuda_for_image
        self.record_stream = DataPrefetcher._record_stream_for_image
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        with torch.cuda.stream(self.stream):
            self.input_cuda()
            self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            self.record_stream(input)
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target

    def _input_cuda_for_image(self):
        self.next_input = self.next_input.cuda(non_blocking=True)

    @staticmethod
    def _record_stream_for_image(input):
        input.record_stream(torch.cuda.current_stream())