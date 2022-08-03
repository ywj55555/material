from libtiff import TIFF
import numpy as np
from utils.os_helper import mkdir
import cv2 as cv
from scipy import misc
import os
import warnings
warnings.filterwarnings("ignore")

def tiff_to_read(tiff_image_name):
    tif = TIFF.open(tiff_image_name, mode = "r")
    im_stack = []
    for im in list(tif.iter_images()):
        im_stack.append(im)
        print(im.shape)
    return im_stack

root_path = r'/home/cjl/ssd/dataset/HIK/need1/'
save_path = r'/home/cjl/ssd/dataset/HIK/nine_channel/'
mkdir(save_path)
img_path = '/home/cjl/ssd/dataset/HIK/nisha/'
width = 682
height = 682
channels = 9
wavelength = [548 ,585 ,627 ,665 ,708 ,746 ,789 ,828 ,0 ]
pixelnum = 9*682*682
tiff = 'nisha_20220712_171937_960_01715_channels.tiff'
tif = TIFF.open(img_path+'/'+tiff, mode='r')
image = tif.read_image()

# # 读取方式2
# imagenp = np.fromfile(root_path+tiff, dtype=np.uint16, count=pixelnum)
# imagenp = imagenp.reshape(682, 682, 9)

# 保存每一个通道图像
for i in range(9):
    cv.imwrite(save_path+tiff[:-5]+'_'+str(i)+'.png',cv.normalize(image[:,:,i],None,0,255,cv.NORM_MINMAX,cv.CV_8UC1))




