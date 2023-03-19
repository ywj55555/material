import numpy as np
import sys
sys.path.append('../')
import os
import cv2
from tqdm import tqdm
from utils.os_helper import mkdir
from skimage import io
import random
random.seed(2021)
np.random.seed(2021)

def mask_color_img(img, mask, color=[255, 0, 255], alpha=0.7):
    '''
    在img上的mask位置上画color
    img: cv2 image
    mask: bool or np.where
    color: BGR triplet [_, _, _]. Default: [0, 255, 255] is yellow.
    alpha: float [0, 1].
    Ref: http://www.pyimagesearch.com/2016/03/07/transparent-overlays-with-opencv/
    '''
    out = img.copy()
    img_layer = img.copy()  # 深拷贝，重新开辟内存，操作不影响img，也算浅拷贝？

    if mask.shape[:2] != img.shape[:2]:
        mask = np.rot90(mask)
    # print('shape0:',np.shape(img_layer))
    # print('shape1:', np.shape(mask))
    img_layer[mask] = color
    # plt.imshow(img_layer)
    # plt.show()
    out = cv2.addWeighted(img_layer, alpha, out, 1 - alpha, 0, out)  # 0是gamma修正系数，不需要修正设置为0，out输出图像保存位置
    return out.astype(np.uint8)

def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color

def Hex_to_RGB(hex):
    r = int(hex[1:3],16)
    g = int(hex[3:5],16)
    b = int(hex[5:7], 16)
    rgb = str(r)+','+str(g)+','+str(b)
    # print(rgb)
    return rgb

def get_random_rgb(color_nums = 2):
    color_list = []
    rgb_list = []
    for i in range(color_nums):
        color = randomcolor()
        if color not in color_list:
            color_list.append(color)
    for x in color_list:
        rgb = Hex_to_RGB(x)
        rgb_ = rgb.split(',')
        rgb2 = [int(y) for y in rgb_]
        rgb_list.append(rgb2)
    return rgb_list

from data.dictNew import *
def changeWaterLable(label):
    label[label == 1] = 4
    label[label == 2] = 1
    label[label == 3] = 2
    label[label == 4] = 3

def modifySkinClothLabel(label):
    for i in range(3, 12):
        label[label == i] = 10
def val_label(png_path, label_path, save_path, label_nums=2, labelList=None):
    rgb_list = get_random_rgb(label_nums)
    label_list = os.listdir(label_path)
    for file in tqdm(label_list):
        if file[-4:] != '.png':
            continue
        if labelList is not None:
            if file[:-4] not in labelList:
                continue
        if os.path.exists(save_path + file):
            continue
        print(file)
        if not os.path.exists(png_path + file):
            print(file, "not exist in : ", png_path)
            continue
        imgGt = cv2.imread(png_path + file)
        # img_label = cv2.imread(label_path + file, cv2.IMREAD_GRAYSCALE)
        img_label = io.imread(label_path + file)
        if file[:-4] in extraTest18:
            changeWaterLable(img_label)
            imgLabel_tmp2 = np.zeros([img_label.shape[0] + 10, img_label.shape[1] + 10],
                                     dtype=np.uint8)
            imgLabel_tmp2[5:-5, 5:-5] = img_label
            img_label = imgLabel_tmp2
        if file[:-4] in allWater18:
            img_label[img_label == 1] = 3
        if file[:-4] in allSkinCloth18:
            img_label[img_label == 3] = 0
            # modifySkinClothLabel(img_label)
            # for i in range(3, 12):
            #     img_label[img_label == i] = 10
            # img_label[img_label == 0] = 255
            # img_label[img_label == 10] = 0

        for color_num in range(1, label_nums + 1):
            imgGt = mask_color_img(imgGt, mask=(img_label == color_num), color=rgb_list[color_num - 1], alpha=0.7)
        cv2.imwrite(save_path + file, imgGt)

def verify_path(path):
    if path.find('\\') == -1:
        return path
    return path.replace('\\', '/')

if __name__ == '__main__':
    # pass
    # rgb_list = get_random_rgb(4)
    from data.dictNew import *
    labelpath = '/home/cjl/dataset_18ch/label/'
    rgbpath = '/home/cjl/dataset_18ch/raw_data/'
    # labelpath = r'/home/cjl/waterDataset/dataset/hefei/label/'
    labelpath = verify_path(labelpath)
    # print(labelpath)
    # labelpath = r'D:\ZY2006224YWJ\material-extraction\muli-spactral-data\07-14-hz\label/'
    pngpath ='/home/cjl/dataset_18ch/needTestRaw/'
    pngpath = verify_path(pngpath)
    # png_path = r'D:/ZY2006224YWJ/spectraldata/water_skin_rgb/'
    # waterLabelPath = r'D:/ZY2006224YWJ/spectraldata/trainLabelAddWater/'
    save_path = r'/home/cjl/dataset_18ch/needTest_rend/'
    save_path = verify_path(save_path)
    mkdir(save_path)
    val_label(pngpath, labelpath, save_path, 4, extraTest18)


    # png_path = r'D:\ZY2006224YWJ\material-extraction\needMark' + '\\'
    # label_path = r'D:\ZY2006224YWJ\material-extraction\needMark\backup\finalLabel' + '\\'
    # test_batch = 4
    # file_list = os.listdir(png_path)
    # file_list = [x for x in file_list if x[-4:] == '.png']
    # print(len(file_list))
    # result_dir = r'D:\ZY2006224YWJ\material-extraction\needMark\labelVal' + '\\'
    # mkdir(result_dir)
    # cnt = math.ceil(len(file_list) / test_batch)
    # color_class = [[0, 0, 255], [255, 0, 0], [0, 255, 0], [128, 128, 128]]
    # for i in range(cnt):
    #     file_tmp = file_list[i * test_batch:(i + 1) * test_batch if (i + 1) < cnt else len(file_list)]
    #     for png_i in range(len(file_tmp)):
    #         print(file_tmp[png_i])
    #         imgGt = cv2.imread(png_path + file_tmp[png_i])
    #         if not os.path.exists(label_path + file_tmp[png_i]):
    #             continue
    #         img_label = cv2.imread(label_path + file_tmp[png_i], cv2.IMREAD_GRAYSCALE)
    #         for color_num in [1, 2, 3, 4]:
    #             print(img_label.shape)
    #             print((img_label == color_num).shape)
    #             imgGt = mask_color_img(imgGt, mask=(img_label == color_num), color=color_class[color_num - 1],
    #                                    alpha=0.7)
    #         cv2.imwrite(result_dir + file_tmp[png_i][:-4] + '_gt.jpg', imgGt)