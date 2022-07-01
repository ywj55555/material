import cv2
import numpy as np
from skimage import io
import os
from utils.os_helper import mkdir

skinLabelPath = 'D:/ZY2006224YWJ/spectraldata/trainLabel/'
waterPath = 'D:/ZY2006224YWJ/spectraldata/label/waterLabel/'
finalPath = 'D:/ZY2006224YWJ/spectraldata/trainLabelAddWater/'

mkdir(finalPath)

labels = os.listdir(skinLabelPath)
for file in labels:
    skin = io.imread(skinLabelPath + file)
    water = io.imread(waterPath + file)
    finalLabel = skin.copy()
    mask = (water == 1) & (skin == 0)
    finalLabel[mask] = 3
    cv2.imwrite(finalPath + file, finalLabel)
