import numpy as np
from tqdm import tqdm
import os
# from spectral import envi
# from skimage import io
# import spectral.io.envi as envi
import cv2
# import math
import torch
from utils.os_helper import mkdir
import pytorch_colors as colors
from utils.yuv2rgb_fun import rgb_to_yuv

import copy #导入标准库中的copy模块


raw_path = r'E:\raw_file\10-11-outdoor'+'\\'
# save_rgbnir_path = r'D:\raw_file\video_4rgbnir'+'\\'
save_rgbnir_path = r'E:\raw_file\10-11-outdoor_space_spectral_nor'+'\\'

# white_noise = r'E:\raw_file\getImage_data1018\white_noise'

# 创建保存文件夹
mkdir(save_rgbnir_path)

raw_list = os.listdir(raw_path)
for raw_envi in tqdm(raw_list):
    if raw_envi!='1633938984.raw':
        continue
    # print()
    raw_data = np.fromfile(raw_path + raw_envi, dtype=np.float32)
    raw_data = raw_data.reshape(18,1020,1020)
    raw_data = raw_data.transpose(1,2,0)
    # 异常值处理
    raw_data[np.isnan(raw_data)]=0
    raw_data[np.where(raw_data<0)]=0

    raw_data_space = raw_data.copy()
    # raw_data = raw_data/np.max(raw_data,axis=0,keepdims=True)
    raw_data = raw_data / np.max(raw_data, axis=2, keepdims=True)

    max_data = np.max(raw_data_space,axis=0,keepdims=True)
    max_data = np.max(max_data,axis=1,keepdims=True)
    raw_data_space = raw_data_space /max_data

    png_r = raw_data[:,:,10]
    png_g = raw_data[:, :, 7]
    png_b = raw_data[:, :, 1]
    nir_png = raw_data[:, :, 16]
    # nir_png = nir_png*255
    png_r = cv2.normalize(png_r,None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8UC1)
    png_g = cv2.normalize(png_g, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    png_b = cv2.normalize(png_b, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    nir_png = cv2.normalize(nir_png, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    rgb_png = np.stack([png_b, png_g, png_r], axis=2)

    cv2.imwrite(save_rgbnir_path + raw_envi[:-4] + '_rgb_spec_nor.png', rgb_png)
    cv2.imwrite(save_rgbnir_path + raw_envi[:-4] + '_nir_spec_nor.png', nir_png)

    png_r2 = raw_data_space[:, :, 10]
    png_g2 = raw_data_space[:, :, 7]
    png_b2 = raw_data_space[:, :, 1]
    nir_png2 = raw_data_space[:, :, 16]
    # png_r2 = cv2.normalize(png_r2, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    # png_g2 = cv2.normalize(png_g2, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    # png_b2 = cv2.normalize(png_b2, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    # nir_png2 = cv2.normalize(nir_png2, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)



    rgb_png2 = np.stack([png_b2,png_g2,png_r2],axis=2)

    cv2.imwrite(save_rgbnir_path+raw_envi[:-4]+'_rgb_space_nor.png',rgb_png2*255)
    cv2.imwrite(save_rgbnir_path+raw_envi[:-4]+'_nir_space_nor.png',nir_png2*255)

    # yuv_tesor = np.expand_dims(rgb_png2, axis=0)
    # yuv_tesor = yuv_tesor.transpose(0,3,1,2)
    # yuv_tesor = torch.tensor(yuv_tesor)
    # yuv = colors.rgb_to_yuv(yuv_tesor).squeeze()
    # yuv = yuv.permute(1,2,0).numpy()
    # yuv2 = rgb_to_yuv(yuv_tesor).squeeze()
    # yuv2 = yuv2.permute(1,2,0).numpy()
    #
    #
    #
    # cv2.imshow("ori", rgb_png2)
    # cv2.imshow("yuv-pytorch", yuv[:,:,0])
    # cv2.imshow("yuv-pytorch1", yuv[:, :, 1])
    # cv2.imshow("yuv-pytorch2", yuv[:, :, 2])
    # cv2.imshow("yuv-my", yuv2[:,:,0])
    # cv2.imshow("yuv-my1", yuv2[:, :, 1])
    # cv2.imshow("yuv-my2", yuv2[:, :, 2])

    # cv2.imshow("img2", img2)
    # cv2.waitKey()
    break
    # print('write')
