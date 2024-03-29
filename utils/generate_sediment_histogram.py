from libtiff import TIFF
import numpy as np
from utils.os_helper import mkdir
import matplotlib.pyplot as plt
import cv2 as cv
from scipy import misc
from scipy import stats
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
save_path = r'/home/cjl/ssd/dataset/HIK/'
img_path = '/home/cjl/ssd/dataset/HIK/nisha/'
# mkdir(save_path)
width = 682
height = 682
channels = 9
wavelength = [548 ,585 ,627 ,665 ,708 ,746 ,789 ,828 ,0 ]
pixelnum = 9*682*682

# 正确显示中文和负号
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# point = [365, 394]
# point = [397, 322]
point = [355, 390]
density_list = []
tifflist = os.listdir(root_path)
tifflist = tifflist[1:]
patch = 8
for tiff in tifflist:
    if tiff[-3:] != 'png':
        continue
    tif = TIFF.open(img_path + tiff[:-4] + '.tiff', mode='r')
    image = tif.read_image()
    density_image = np.mean(image[:, :, :3], axis=2)
    density = np.mean(density_image[point[0] - patch: point[0] + patch, point[1] - patch:point[1] + patch])
    ndwi_image = (image[:, :, 0] - image[:, :, 7]) / (image[:, :, 0] + image[:, :, 7])
    ndwi = np.mean(ndwi_image[point[0] - patch: point[0] + patch, point[1] - patch:point[1] + patch])
    # density = density * ndwi
    density_list.append(density)
    # 画图，plt.bar()可以画柱状图
print(density_list)
density_list = density_list / np.max(np.array(density_list))
# density_list = density_list[::-1]

x_list = [x for x in range(len(density_list))]
# for i in range(len(x_list)):
#     print(x_list[i], density_list[i])
#     plt.bar

# 设置图片名称 Divided ndwi Multiplied ndwi
# plt.title("Sediment concentration Multiplied ndwi histogram")
# plt.title("泥沙")
# # 设置x轴标签名
plt.xlabel("实验轮次")
# # 设置y轴标签名
plt.ylabel("相关统计量") # intensity concentration
# plt.ylim(2000,3500)
# 显示
plt.scatter(x_list, density_list)

sta = stats.pearsonr(x_list, density_list)
print(sta)

parameter = np.polyfit(x_list, density_list, 1)
p = np.poly1d(parameter)
plt.ylim(0, 1)
density_list_fitting = p(x_list)
plt.plot(x_list, density_list_fitting, color='g')

diff = density_list - density_list_fitting
diff2 = diff**2
error_var = np.mean(diff2)
print(error_var)
plt.savefig(save_path + 'three_band.png')
plt.show()

    # cv.imwrite(save_path + tiff[:-5] + '.png',
    #            cv.normalize(image[:, :, 0], None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1))

    # 保存每一个通道图像
    # for i in range(9):
    #     cv.imwrite(save_9png_path + tiff[:-5] + '_' + str(i) + '.png',
    #                cv.normalize(image[:, :, i], None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1))

    # 保存伪彩色图像
    # png_b = cv.normalize(image[:, :, 0], None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
    # png_g = cv.normalize(image[:, :, 6], None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
    # png_r = cv.normalize(image[:, :, 8], None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
    # rgb = np.stack([png_b, png_g, png_r], axis=2)
    # cv.imwrite(save_rgb_path + tiff[:-5] + '_rgb' + '.png', rgb)



