import os

import cv2

tifpath = 'D:/dataset/shuichi/rgb/'

tiflists = os.listdir(tifpath)

for tif in tiflists:
    if tif[-4:] != '.tif':
        continue
    os.rename(tifpath + tif, tifpath + tif[:-4] + '.png')