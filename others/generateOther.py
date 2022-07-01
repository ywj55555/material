import cv2
from skimage import io
import os

pngpath = 'D:/ZY2006224YWJ/spectraldata/RiverSkinDetection3/'

labelpath = 'D:/ZY2006224YWJ/spectraldata/label/.imageLabelingSessionwater3_SessionData/'

dstLabel = '20220623144115595'
dstLabelData = io.imread(labelpath + '20220623144115595' + '.png')

pngs = os.listdir(pngpath)

for png in pngs:
    if os.path.exists(labelpath + png) or png[-4:] != '.png':
        continue

    cv2.imwrite(labelpath + png, dstLabelData)