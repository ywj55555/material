import random
import gc
from skimage import io
import math
import cv2 as cv
import cv2
import numpy as np
from data.dictNew import *
from utils.load_spectral import *
import os
import copy
from sklearn.preprocessing import LabelEncoder
from torch.autograd import Variable
import torch
from multiprocessing import Process,Pool
import multiprocessing as mp

code2label = [0,2,2,2,0,2,2,2,1,0,0,0,0,0,0,3,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0] #1:skin 2:cloth 3:plant 0:other
# label2class = [255,1,2,0]
label2class = [255,1,0]
label2target = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

SKIN_TYPE = 0
CLOTH_TYPE = 1
PLANT_TYPE = 2

# DATA_TYPE = SKIN_TYPE
# DATA_TYPE = CLOTH_TYPE
DATA_TYPE = PLANT_TYPE

SELECT_ALL = 0  # select all pixels for training and testing
SELECT_RANDOM = 1  # select pixels randomly for training and testing

waterLabelPath = '/home/glk/datasets/Water_shenzhen/label/'
waterImgRootPath = '/home/glk/datasets/Water_shenzhen/'

# 每一类 取样最小间隔
min_interval = [4,10]
# 记录每张图 每一个类 取样了多少patch
class_nums_record = {1:0,2:0}
# output_log = open('./log/output_generate.log','w')

def generateData(train_set, dataType, num, length, selectMode=SELECT_RANDOM,nora = True, class_nums = 2, intervalSelect = False, featureTrans = False):
    Data = []
    Label = []
    if dataType == "train":
        # dataFile = SeaFile
        data_file = "/home/glk/datasets/Water_shenzhen/train" + train_set + ".txt"
    else:
        # dataFile = waterFileTest
        data_file = "/home/glk/datasets/Water_shenzhen/test" + train_set + ".txt"
    labelpath = waterLabelPath
    imgpath = waterImgRootPath

    f = open(data_file,'r')
    dataFile = f.readlines()
    num_count = 0
    for file in dataFile:
        file = file.split('\n')[0]
        # 以rgb开头
        # t1 = time.time()
        # 读取标签文件
        # 1 skin 2 cloth 3 other
        imgLabel = io.imread(labelpath + file + '.png')
        # 读取img文件
        imgData = None
        if os.path.exists(imgpath + file[3:] + '.img'):
            imgData = envi_loader(imgpath, file[3:], False)
        # cv2.imwrite('./test.jpg',(imgData[:,:,:]).mean(2)*255/imgData[:,:,:].mean(2).max())#
        # 没必要 特征变换 增加之前设计的斜率特征
        # np.save(imgpath + file[3:] + '.npy',imgData)
        if imgData is None:
            print("Not Found ",file)
            continue
        # H W 22
        # if featureTrans:
        #     print("kindsOfFeatureTransformation......")
        #     # 11 -》 21
        #     imgData = kindsOfFeatureTransformation(imgData, nora)
        # t4 = time.time()
        # print(imgData.shape)
        # if imgData.shape != (1415, 1859, 22):
        #     continue
        # print(typeCode)
        # 11*11像素块 类别预测 作为中心像素类别 可以利用上下文信息 提升准确率
        # 分割出像素块 返回像素块和对应的类别
        if intervalSelect:
            pix, label = intervalGenerateAroundDeltaPatchAllLen(num_count, imgData, imgLabel, num, length, file, 50 ,
                                                        class_nums)
        else:
            pix, label = generateAroundDeltaPatchAllLen(num_count, imgData, imgLabel, num, length,class_nums)
        #print(pix.shape)
        #print(label.shape)
        # t5 = time.time()
        # print('read label:', t2 - t1, ',read envi:', t3 - t2, ',transform:', t4 - t3, ',genrate:', t5 - t4)
        # read label: 3.676800489425659 ,read envi: 9.025498867034912 ,transform: 0.22420549392700195 ,genrate: 2.00569748878479
        Data.extend(pix)
        Label.extend(label)
        num_count += 1
    f.close()
    Data = np.array(Data, dtype=np.float32)
    Label = np.array(Label, dtype=np.int8)

    return Data, Label

def convert_to_one_hot(y, C):
    return np.eye(C,dtype=np.int8)[y.reshape(-1)]


# 随机选取 存在的问题 选取可能不均匀 可能边缘，或者一些 比较分歧、重要的地方不能保证选取到，数据多样性就不能保证
def generateAroundDeltaPatchAllLen(num_count, imgData, imgLabel, labelCount=2000, length=11,class_nums = 2, overexposure_thre = 45000):
    row, col, d = imgData.shape
    imgData = np.array(imgData)
    # H W C -> C H W
    print("move overexposure area....")
    max_mask = np.max(imgData, axis=2)
    imgLabel[max_mask > overexposure_thre] = 0
    img = imgData.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    # del max_mask
    gc.collect()
    # img.shape -> (1, channels, row, col)
    # 像素块变量 保存提取的像素块
    # pix = np.empty([0, d, length, length], dtype=float)
    ##########
    pix = np.empty([0, 3], dtype=float)
    labels = np.empty([0, class_nums])
    length1 = int(length / 2)
    for label in range(1, class_nums + 1):  # 共有3类，取值为1、2、3、...、30，0为未标注
        label_index = np.where(imgLabel == label)  # 找到标签值等于label的下标点 size：2*个数
        pixels_nums = len(label_index[0])
        if pixels_nums == 0:
            continue
        delete_index = []
        for i in range(pixels_nums):
            if (label_index[0][i] <= length1 or label_index[0][i] >= row - length1 or label_index[1][i] <= length1 or
                    label_index[1][i] >= col - length1):
                delete_index.append(i)
        # 删除多列
        label_index = np.delete(label_index, delete_index, axis=1)
        pixels_nums = len(label_index[0])
        if pixels_nums == 0:
            continue
        # tmp = np.array(label2target[code2label[label]])
        # 获取对应类别 one-hot 标签
        tmp = np.array(convert_to_one_hot(np.array([class_order for class_order in range(class_nums)]), class_nums)[
                           label2class[label]])

        # 以下 用于 不均衡采样 类别样本
        per_class_num = [labelCount]
        per_class_num_care = [labelCount for _ in range(class_nums-1)]
        per_class_num.extend(per_class_num_care)
        # print(per_class_num)
        # [1000,2000]
        # per_class_num.append(l)
        # selectMode = SELECT_RANDOM
        # if selectMode == SELECT_ALL:
        #     for index in range(pixels_nums):
        #         labels = np.append(labels, np.reshape(tmp, [-1, class_nums]), axis=0)
        #         x = label_index[0][index]
        #         y = label_index[1][index]
        #         pixAround = img[:, :, x - length1: x + length1 + 1, y - length1: y + length1 + 1]  # 3*3*4
        #         pix = np.append(pix, pixAround, axis=0)
        # elif selectMode == SELECT_RANDOM:
        for num in range(per_class_num[label2class[label]]):
            index = random.randint(0, pixels_nums - 1)  # 前闭后闭区间 随机选取
            # labels = np.append(labels, label)
            #print(np.reshape(tmp, [-1, 4]), np.reshape(tmp, [-1, 4]).shape)
            labels = np.append(labels, np.reshape(tmp, [-1, class_nums]), axis=0)
            x = label_index[0][index]
            y = label_index[1][index]
            # B C H W
            # if length % 2 == 0:
            #     pixAround = img[:, :, x - length1: x + length1, y - length1: y + length1]  # 3*3*4
            # else:
            #     pixAround = img[:, :, x - length1: x + length1 + 1, y - length1: y + length1 + 1]  # 3*3*4
            # pixAround = label_index[:][index:index+1]
            pixAround = [[1,1,1]]
            pixAround[0][0] = x
            pixAround[0][1] = y
            pixAround[0][2] = num_count
            # 堆叠像素块
            pix = np.append(pix, pixAround, axis=0)
    labels = np.array(labels).astype('int64')
    pix = np.array(pix, dtype=float)
    shuffle_ix = np.random.permutation(np.arange(labels.shape[0]))
    pix = pix[shuffle_ix]
    labels = labels[shuffle_ix]
    return pix, labels

def intervalGenerateAroundDeltaPatchAllLen(num_count, imgData, imgLabel, labelCount=2000, length=11, filename = 'null',thre=500, class_nums = 2, overexposure_thre = 45000):
    row, col,d = imgData.shape
    # img = np.expand_dims(imgData, 0)
    ####################imgData = np.array(imgData)
    # H W C -> C H W
    # 去掉过曝区域
    print("move overexposure area....")
    max_mask = np.max(imgData, axis=2)
    imgLabel[max_mask > overexposure_thre] = 0
    ####################
    img = imgData.transpose(2, 0, 1)
    del max_mask
    gc.collect()
    # B C H W
    img = np.expand_dims(img, 0)
    length1 = int(length / 2)
    # pix = np.empty([0, d, length, length], dtype=float)
    ##############################
    pix = np.empty([0, 3], dtype=float)
    labels = np.empty([0, class_nums])
    # 去掉过曝区域

    for label in range(1,class_nums + 1):  #label 已经转为0，1，2，3，4了 ,3植物 2衣物 1皮肤 4其他，0无标签
        # cnt = 1
        label_index = np.where(imgLabel[length1:-length1,length1:-length1] == label)  # 找到标签值等于label的下标点 size：2*个数
        pixels_nums = len(label_index[0])
        if pixels_nums <= thre:
            continue
        '''方案1
            1、如果 pixels_nums 小于 thre，说明目标过小，为无效数据，不采样；
            2、如果 pixels_nums < 4*labelCount ,则间隔行列采样 pixels_nums 不管数量；保证样本的稀疏性
            3、否则：
            计算pixels_nums是labelCount的多少倍，然后开立方根，向下取整，得到m，
            对label_index 进行按照行序号排序，间隔m-1采样 （从序号 m-1开始采样），开始序号为 (新的pixels_nums长度取模m +m-1)//2 (肯定小于m)
            如果满足，则结束，否则：
            对label_index 进行按照列序号排序，间隔m-1采样
            然后余下的为采样索引
            方案2-保证样本多样性(间隔取样)，根据该类数目选择间隔，设置一个最低和最高间隔
                在以上基础上尽可能保证数量均衡
                间隔取样：边界部分取样点可能相邻
                边界部分本来就难识别，多取样也并非坏事
        '''
        #最小采样间隔
        interval =  min_interval[label-1]
        #间隔1采样，数量除以4
        if pixels_nums > interval*interval*labelCount:
            mul = math.sqrt(pixels_nums/labelCount)
            interval = round(mul)
        print('before :',filename, 'label:',label, 'interval ',interval,'before :',pixels_nums)
        # print(label)
        # 也可以直接在原数组上操作，就不用排序索引了 10520没有进行采样
        # 一定要用深拷贝 可以了解一下 深拷贝 和 浅拷贝
        imgLabel_tmp = copy.deepcopy(imgLabel)
        # imgLabel_tmp = imgLabel
        # 隔行取样
        # cnt = imgLabel.shape[0]//interval
        cnt = math.ceil(imgLabel.shape[0] / interval)
        # print('cnt: ',cnt)
        mask = []
        for i in range(cnt):
            mask += [mk for mk in range(i*interval,interval*(i+1)-1 if interval*(i+1)-1<imgLabel.shape[0] else imgLabel.shape[0])]
            # imgLabel_tmp[i*cnt:cnt*(i+1)-1,:]=0
        # print('mask',mask)
        imgLabel_tmp[mask,:] = 0
        new_label_index = np.where(imgLabel_tmp == label)
        print('after row sampling: ',len(new_label_index[0]))
        # print(imgLabel_tmp)
        # 隔列取样
        mask = []
        # print(filename, label, interval)
        # cnt = imgLabel.shape[1] // interval
        cnt = math.ceil(imgLabel.shape[1] / interval)
        # print('cnt: ', cnt)
        for i in range(cnt):
            mask += [mk for mk in range(i*interval,interval*(i+1)-1 if interval*(i+1)-1<imgLabel.shape[1] else imgLabel.shape[1])]
        imgLabel_tmp[:,mask] = 0
        # print('mask', mask)
        #取样后的索引
        new_label_index = np.where(imgLabel_tmp[length1:-length1,length1:-length1] == label)

        #删掉边界像素！！
        # new_pixels_nums = len(new_label_index[0])
        # # 还要删掉边界位置
        # new_delete_index = []
        # for i in range(new_pixels_nums):
        #     if (new_label_index[0][i] <= length1 or new_label_index[0][i] >= row - length1 or new_label_index[1][
        #         i] <= length1 or
        #             new_label_index[1][i] >= col - length1):
        #         new_delete_index.append(i)
        # # 删除多列
        # new_label_index = np.delete(new_label_index, new_delete_index, axis=1)
        #取样所有label_index

        new_count = len(new_label_index[0])
        print('after sampling colounm',filename,label,new_count)
        # print(imgLabel_tmp)
        # continue
        # f.write(filename+' sampling '+str(label)+ ' class: '+str(new_count)+'\n')
        tmp = np.array(convert_to_one_hot(np.array([class_order for class_order in range(class_nums)]), class_nums)[
                           label2class[label]])
        # print(tmp)
        # print(label)

        global class_nums_record
        class_nums_record[label] += new_count
        for index in range(new_count):
            labels = np.append(labels, np.reshape(tmp, [-1, class_nums]), axis=0)
            x = new_label_index[0][index]
            y = new_label_index[1][index]
            # pixAround = img[:,:,x : x + 2*length1 + 1, y : y + 2*length1 + 1]  # 21,21,bands
            # pixAround = new_label_index[:][index]
            pixAround = [[1,1,1]]
            pixAround[0][0] = x+length1
            pixAround[0][1] = y+length1
            pixAround[0][2] = num_count
            # if label != imgLabel[x+length1,y+length1]:
            #     print('################################')
            # if labels[0,label-1] != 1:
            #     if label != 1:
            #         print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            #         print(label)
            # if label == 1:
            #     print(labels)

                # print(labels)
            pix = np.append(pix, pixAround, axis=0)
            # np.save(save_path + filename + name_list[label - 1] + str(index).rjust(4, '0') + '.npy', pixAround)
    labels = np.array(labels).astype('int64')
    pix = np.array(pix)
    # print('!!!!!!!!!!!!!!!!!!!!!!!')
    # print(pix.shape)
    # print('label shape:',labels.shape)
    # 这个不是 单张图 的抽样结果吧！！！
    shuffle_ix = np.random.permutation(np.arange(labels.shape[0]))
    pix = pix[shuffle_ix]
    labels = labels[shuffle_ix]
    print(filename, "sampling :",class_nums_record)
    # B C H W
    return pix, labels



if __name__ == '__main__':
    pass
