import shutil
import os
from data.dictNew import *
from utils.os_helper import mkdir
src = ['20211021140510.png']
srcpath = '/home/cjl/dataset_18ch/rgbAll_rend/'
srcpngpath = '/home/cjl/dataset_18ch/raw_data/'
dstpng = '/home/cjl/dataset_18ch/rgb_rend_5/'
mkdir(dstpng)
for file in src:
    shutil.copy(srcpngpath + file, dstpng + file)
    shutil.copy(srcpath + file, dstpng + file[:-4] + '_rend.png')
# rootpath = 'E:/tmp/lg/'
# waterPath = 'E:/tmp/hzWater/'
# mkdir(waterPath)
# moveFile = waterFile
# pathlist = os.listdir(rootpath)
# pathlist = [x for x in pathlist if x[-3:] == 'rgb']
# cnt = 0
# for png in moveFile:
#     for tmp_path in pathlist:
#         oripath = rootpath + tmp_path + '/' + png + '.tif'
#         dstpath = waterPath + png + '.png'
#         if os.path.exists(oripath):
#             shutil.copy(oripath, dstpath)
#             cnt+=1
#             break
# print(cnt)
# path = '/home/cjl/dataset_18ch/waterBmh/'
# dstpath = '/home/cjl/dataset_18ch/raw_data/'
# files = os.listdir(path)
# for file in files:
#     if file[-4:] != '.png':
#         continue
#     if not os.path.exists(dstpath + file):
#         shutil.copy(path + file, dstpath + file)

