from scipy import misc
import cv2
import os
from skimage import io



doc_dir = '/home/glk/datasets/shenzhen_tiff/20220429/'
files = os.listdir(doc_dir)
name = []
for i in files:
    if i[-5:] == '.tiff':
        # print(i)
        # print(doc_dir + i)
        spec_img = io.imread(doc_dir + i)
        # print(spec_img.shape)
        img = spec_img[:, :, -1]
        max_int = img.max()
        img = img/max_int *255
        # print(max_int)
        cv2.imwrite(doc_dir + 'grey_img/' + i.split('.')[0] + '.png', img)
        print(doc_dir + 'grey_img/' + i.split('.')[0] + '.png')
        name.append(i)
        # break