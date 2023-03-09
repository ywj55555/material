import cv2
import numpy as np
from utils.os_helper import mkdir

"""
x, y, w, h = cv2.boundingRect(img)
    参数：
    img  是一个二值图
    x，y 是矩阵左上点的坐标，
    w，h 是矩阵的宽和高

cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
    img：       原图
    (x，y）：   矩阵的左上点坐标
    (x+w，y+h)：是矩阵的右下点坐标
    (0,255,0)： 是画线对应的rgb颜色
    2：         线宽
"""


def erode_demo(image, kernelsize):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernelsize, kernelsize))
    dst = cv2.erode(image, kernel)
    return dst


def erode_specific_label(image, kernelsize=10, label=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernelsize, kernelsize))
    tmp_img = np.zeros_like(image)
    tmp_img[image == label] = label
    dst = cv2.erode(tmp_img, kernel)
    # erode_area = (dst==label)!=(image==label)
    erode_area = tmp_img != dst
    image[erode_area] = 0
    return image


def erode_all_label(image, kernelsize=10):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernelsize, kernelsize))
    tmp_img = np.zeros_like(image)
    tmp_img[image != 0] = 255
    dst = cv2.erode(tmp_img, kernel)
    erode_area = tmp_img != dst
    image[erode_area] = 0
    return image

def open_all_diff_size(image, erode_kernel, dilate_kernel):
    str_erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_kernel, erode_kernel))
    tmp_img = np.zeros_like(image)
    tmp_img[image != 0] = 255
    dst = cv2.erode(tmp_img, str_erode_kernel)
    str_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel))
    dst = cv2.dilate(dst, str_dilate_kernel)
    erode_area = tmp_img != dst
    image[erode_area] = 0
    return image

def dilate_specific_label(image, kernelsize=10, label=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernelsize, kernelsize))
    tmp_img = np.zeros_like(image)
    # tmp_img2 = np.zeros_like(image)
    tmp_img[image == label] = label
    # tmp_img2[image == label] = label
    dst = cv2.dilate(tmp_img, kernel)
    # erode_area = (dst==label)!=(image==label)
    dilate_area = dst != tmp_img
    image[dilate_area] = label
    return image


def open_specific_label(image, kernelsize=10, label=1):
    dst = erode_specific_label(image, kernelsize, label)
    dst = dilate_specific_label(dst, kernelsize, label)
    return dst


def open_specific_label_difsize(image, erode_size=10, dilate_size=10, label=1):
    dst = erode_specific_label(image, erode_size, label)
    dst = dilate_specific_label(dst, dilate_size, label)
    return dst

# def close_specific_label_difsize(image, erode_size=10, dilate_size=10, label=1):
#     dst = erode_specific_label(image, erode_size, label)
#     dst = dilate_specific_label(dst, dilate_size, label)
#     return dst


# 膨胀
def dilate_demo(image, kernelsize):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernelsize, kernelsize))
    dst = cv2.dilate(image, kernel)
    return dst


# 先腐蚀(瘦)后膨胀(胖)叫开运算（因为先腐蚀会分开物体，这样容易记住），其作用是：分离物体，消除小区域。
def open_demo(image, kernel):
    str_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))  # 定义结构元素
    dst = cv2.morphologyEx(image, cv2.MORPH_OPEN, str_kernel)  # 开运算
    return dst


# 先膨胀后腐蚀（先膨胀会使白色的部分扩张，以至于消除/"闭合"物体里面的小黑洞，所以叫闭运算）
def close_demo(image, kernel):
    str_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))  # 定义结构元素
    dst = cv2.morphologyEx(image, cv2.MORPH_CLOSE, str_kernel)  # 开运算
    return dst


def open_demo_diff_size(image, erode_kernel, dilate_kernel):
    str_erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_kernel, erode_kernel))
    dst = cv2.erode(image, str_erode_kernel)

    str_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel))
    dst = cv2.dilate(dst, str_dilate_kernel)
    return dst


def paint_rect(png, ori, morphology=1, kernel=25, thre=200):
    '''可以增加额外的判断逻辑;
        实际上可以存在嵌套情况：有两个人，一个距离较远，另一个距离较近
        不存在内部嵌套情况，使用cv2.RETR_EXTERNAL即可
    '''
    png = png.astype(np.uint8)
    if morphology == 0:
        pass
    elif morphology == 1:
        png = close_demo(png, kernel)
    elif morphology == 2:
        png = open_demo(png, kernel)
    elif morphology == 3:
        png = open_demo_diff_size(png, kernel, kernel + 25)
    else:
        pass
    contours, hierarchy = cv2.findContours(png, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(0, len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        if w < thre or h < thre:
            continue
        cv2.rectangle(ori, (x, y), (x + w, y + h), (0, 0, 255), 15)
    return ori


if __name__ == '__main__':
    png_path = './rectangle_test.png'
    img = cv2.imread(png_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    res = paint_rect(img, binary)
    # contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # for i in range(0,len(contours)):
    # x, y, w, h = cv2.boundingRect(contours[i])
    # cv2.rectangle(img, (x,y), (x+w,y+h), (153,153,0), 2)
    cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    cv2.imshow('test', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# png_path = r'D:\ZY2006224YWJ\material-extraction\muli-spactral-data\hz0719\res\4\20210719112343003.png'
# png = cv2.imread(png_path)
#
# mask1 = png[:,:,0]==255
# mask2 = png[:,:,2]==255
# mask3 = png[:,:,1]==255
#
# img = np.zeros(png.shape[:2])
# img[mask1]=255
# img[mask2]=255
# img[mask3]=255
# print(png.shape)
#
# cv2.namedWindow("test", cv2.WINDOW_NORMAL)
# cv2.imshow('test',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
