import cv2
import numpy as np
import spectral.io.envi as envi
from utils.load_spectral import envi_loader

path = 'D:/ZY2006224YWJ/material-extraction/spectraldata/'
filename = '20220415104627834'
enviData = envi.open(path + filename + '.hdr', path + filename + '.img')
imgData = enviData.load()
imgData = np.array(imgData)
print(np.max(imgData))
data = np.max(imgData, axis=2)
print(np.max(data))
maskOver = data > 38000
pngdata = np.zeros_like(maskOver,dtype=np.uint8)
pngdata[maskOver] = 255
print(pngdata.shape)
cv2.imwrite('res/overExposureArea.png', pngdata)
