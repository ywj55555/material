import os

import cv2

tifpath = 'D:/ZF2121133HHX/water/daytime/rgb/'

tiflists = os.listdir(tifpath)

for tif in tiflists:
    if tif[-4:] != '.tif':
        continue
    os.rename(tifpath + tif, tifpath + tif[:-4] + '.png')