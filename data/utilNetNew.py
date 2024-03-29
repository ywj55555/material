import gc
import random

import cv2
from skimage import io
import math
from multiprocessing import Process, Pool
import multiprocessing as mp
# import numpy as np
from data.dictNew import *
from utils.load_spectral import *
import os
import copy

# from sklearn.preprocessing import LabelEncoder
# from torch.autograd import Variable
# import torch

code2label = [255, 2, 2, 2, 0, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0]  # 1:skin 2:cloth 3:plant 0:other
# label2class = [255,1,2,0]
# label2class = [0,1,2]
# from waterAndSkinModel128 import class_nums

class_nums = 4
label2class = range(class_nums)
labelToclass18skin = [255, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0]
label2target = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

SKIN_TYPE = 0
CLOTH_TYPE = 1
PLANT_TYPE = 2

# DATA_TYPE = SKIN_TYPE
# DATA_TYPE = CLOTH_TYPE
DATA_TYPE = PLANT_TYPE

SELECT_ALL = 0  # select all pixels for training and testing
SELECT_RANDOM = 1  # select pixels randomly for training and testing
#
# env_data_dir3 = '/home/cjl/dataset/envi/'
# label_data_dir = '/home/cjl/dataset/label/'
video3_img_path = 'D:/ZF2121133HHX/20220407/vedio3/'
video6_img_path = 'D:/ZF2121133HHX/20220407/vedio6/'

video3_label_path = 'D:/ZF2121133HHX/20220407/video3_labels/'
video6_label_path = 'D:/ZF2121133HHX/20220407/video6_labels/'

waterLabelPath = '/home/cjl/ssd/dataset/shenzhen/label/Label_rename/'

waterImgRootPath = '/home/cjl/ssd/dataset/shenzhen/img/train/'
# waterImgRootList = os.listdir(waterImgRootPath)
# waterImgRootList = [x for x in waterImgRootList if x[-4:] == '.img']
waterImgRootPathList = ['vedio1', 'vedio2', 'vedio3', 'vedio4', 'vedio5', 'vedio6', 'vedio7']
skinAndWaterLabelPath = 'D:/ZY2006224YWJ/spectraldata/trainLabelAddWater/'

all_label_path = '/home/cjl/dataset_18ch/label/'
all_png_path = '/home/cjl/dataset_18ch/raw_data/'
skinClothRawPath = '/home/cjl/dataset_18ch/raw_data/'
waterRawPath = '/home/cjl/dataset_18ch/waterBmh/'

# label_data_dir_6se = '/home/cjl/data/sensor6/label/'
# tif_dir = '/home/cjl/data/sensor6/tif_data/'
# env_data_dir = 'E:/BUAA_Spetral_Data/hangzhou/envi/'
# label_data_dir = 'E:/BUAA_Spetral_Data/hangzhou/label/'

# 每一类 取样最小间隔
class_nums18 = 10
min_interval = [2 for _ in range(class_nums18 + 1)]

# 记录每张图 每一个类 取样了多少patch
# class_nums_record = {0:0, 1:0,2:0}
class_nums_record = {x: 0 for x in range(class_nums18 + 1)}


# output_log = open('./log/output_generate.log','w')

def modifyWaterLabel(label):
    label[label == 0] = 10
    label[label == 1] = 3


def modifySkinClothLabel(label):
    for i in range(3, 12):
        label[label == i] = 10


def generateData(dataType, num, length, typeCode, selectMode=SELECT_RANDOM, nora=True, class_nums=2,
                 intervalSelect=True, featureTrans=True):
    Data = []
    Label = []
    if dataType == 'sea':
        dataFile = SeaFile
        labelpath = waterLabelPath
        imgpath = waterImgRootPath
    # if dataType == 'train':
    #     dataFile = trainFile
    # elif dataType == 'test':
    #     dataFile = testFile
    # if dataType == 'water':
    #     dataFile = waterFile
    #     labelpath = waterLabelPath
    #     imgpath = waterImgRootPath
    # elif dataType == 'water':
    #     dataFile = waterFile
    #     imgpath = waterImgRootPath
    #     labelpath = waterLabelPath
    else:
        dataFile = waterFileTest
        labelpath = waterLabelPath
        imgpath = waterImgRootPath
    for file in dataFile:
        print(file)
        # 以rgb开头
        # t1 = time.time()
        # 读取标签文件
        # 1 skin 2 cloth 3 other
        imgLabel = io.imread(labelpath + file + '.png')
        # t2 = time.time()
        # imgData = envi_loader(os.path.join(env_data_dir,file[:8])+'/', file,nora)
        # 读取img文件
        imgData = None
        if os.path.exists(imgpath + file[3:] + '.img'):
            imgData = envi_loader(imgpath, file[3:], False)
        else:
            for tmpImgPath in waterImgRootPathList:
                if os.path.exists(imgpath + tmpImgPath + '/' + file[3:] + '.img'):
                    imgData = envi_loader(imgpath + tmpImgPath + '/', file[3:], False)
                    break
        # t3 = time.time()
        # 没必要 特征变换 增加之前设计的斜率特征
        if imgData is None:
            print("Not Found ", file)
            continue
        # H W 22
        if featureTrans:
            print("kindsOfFeatureTransformation......")
            # 11 -》 21
            imgData = kindsOfFeatureTransformation(imgData, nora)
        else:
            if nora:
                print("normalizing......")
                imgData = envi_normalize(imgData)

        # t4 = time.time()
        # print(imgData.shape)
        # if imgData.shape != (1415, 1859, 22):
        #     continue
        # print(typeCode)
        # 11*11像素块 类别预测 作为中心像素类别 可以利用上下文信息 提升准确率
        # 分割出像素块 返回像素块和对应的类别
        if intervalSelect:
            pix, label, _ = intervalGenerateAroundDeltaPatchAllLen(imgData, imgLabel, num, length, file, 50,
                                                                class_nums)
        else:
            pix, label, _ = generateAroundDeltaPatchAllLen(imgData, imgLabel, typeCode, num, length, selectMode,
                                                        class_nums)
        # print(pix.shape)
        # print(label.shape)
        # t5 = time.time()
        # print('read label:', t2 - t1, ',read envi:', t3 - t2, ',transform:', t4 - t3, ',genrate:', t5 - t4)
        # read label: 3.676800489425659 ,read envi: 9.025498867034912 ,transform: 0.22420549392700195 ,genrate: 2.00569748878479
        Data.extend(pix)
        Label.extend(label)
    return Data, Label


def singleProcessGenerateData(dataType, dataFile, num, length, bands, activate, nora=False, class_nums=2,
                              intervalSelect=True, featureTrans=True, addSpace=False):
    id = os.getpid()
    print("当前进程id: ", id, "begin!!!!!!")
    Data = []
    DataSpace = []
    Label = []
    if dataType == 'sea':
        labelpath = waterLabelPath
        imgpath = waterImgRootPath
    elif dataType == 'bmhWater':
        labelpath = '/home/cjl/dataset_18ch/WaterLabel_mask_221011/'
        imgpath = '/home/cjl/dataset_18ch/waterBmh/'
    elif dataType == 'RiverSkinDetection1':
        labelpath = skinAndWaterLabelPath
        imgpath = 'D:/ZY2006224YWJ/spectraldata/water_skin/'
    elif dataType == 'draft':
        labelpath = '/home/cjl/ssd/dataset/HIK/shuichi_label/'
        imgpath = '/home/cjl/ssd/dataset/HIK/shuichi/img/'
        print(labelpath)
        print(imgpath)
    elif dataType == 'HZRiverSkinClothTrain' or dataType == 'HZRiverSkinClothTest':
        labelpath = '/home/cjl/spectraldata/trainLabelAddWater/'
        imgpath = '/home/cjl/spectraldata/RiverLakeTrainData/'
    elif dataType == 'allTrain18' or dataType == 'allTest18':
        labelpath = all_label_path
        imgpath = None
    else:
        labelpath = skinAndWaterLabelPath
        imgpath = 'D:/ZY2006224YWJ/spectraldata/water_skin/'
    dataFile_LEN = len(dataFile)
    for file_order, file in enumerate(dataFile):
        print(file)
        print(file_order, "/", dataFile_LEN)
        # output_log.writelines(str(file_order)+ "/" + str(dataFile_LEN))
        # 1 skin 2 cloth 3 other
        imgLabel = io.imread(labelpath + file + '.png')
        if file in allWater18:
            modifyWaterLabel(imgLabel)
        if file in allSkinCloth18:
            modifySkinClothLabel(imgLabel)
        # t2 = time.time()
        imgData = None
        if dataType == dataType == 'allTrain18' or dataType == 'allTest18':
            cut_num = 10
            overexposure_thre = 1000
            if os.path.exists(skinClothRawPath + file + '.raw'):
                imgData = raw_loader(skinClothRawPath, file, nora, cut_num=cut_num)
            elif os.path.exists(waterRawPath + file + '.raw'):
                imgData = raw_loader(waterRawPath, file, nora, cut_num=cut_num)
            else:
                print(file, ' raw not exist!!!')
            imgLabel = imgLabel[cut_num:-cut_num, cut_num:-cut_num]
        else:
            overexposure_thre = 45000
            if os.path.exists(imgpath + file + '.img'):
                imgData = envi_loader(imgpath, file, bands, False)
            else:
                print(file, 'not found!!!')
                continue
        # t3 = time.time()
        # 没必要 特征变换 增加之前设计的斜率特征
        # H W 22
        if featureTrans:
            print("kindsOfFeatureTransformation......")
            # output_log.writelines("kindsOfFeatureTransformation......")
            # 11 -》 21
            imgData = kindsOfFeatureTransformation_slope(imgData, activate, nora)
        else:
            if nora:
                print("normalizing......")
                # output_log.writelines("normalizing......")
                imgData = envi_normalize(imgData)
        # 11*11像素块 类别预测 作为中心像素类别 可以利用上下文信息 提升准确率
        # 分割出像素块 返回像素块和对应的类别
        if intervalSelect:
            print("intervalGenerateAroundDeltaPatchAllLen...")
            pix, label, pixSpace = intervalGenerateAroundDeltaPatchAllLen(imgData, imgLabel, num, length, file, 50, class_nums,
                                                                overexposure_thre=overexposure_thre, addSpace=addSpace)
        else:
            print("random sampling...")
            pix, label, pixSpace = generateAroundDeltaPatchAllLen(imgData, imgLabel, num, length, class_nums,
                                                        overexposure_thre=overexposure_thre, addSpace=addSpace)
        Data.extend(pix)
        if addSpace:
            DataSpace.extend(pixSpace)
        Label.extend(label)
        del imgData
        gc.collect()
    # id = os.getpid()
    print("当前进程id: ", id, "done!!!!!!")
    return Data, Label, DataSpace


def singleProcessGenerateRgbData(dataFile, num, rgbpath, labelpath, length, class_nums=2, intervalSelect=True):
    id = os.getpid()
    print("当前进程id: ", id, "begin!!!!!!")
    Data = []
    Label = []
    # if dataType == 'sea':
    #     labelpath = waterLabelPath
    #     rgbpath = waterImgRootPath
    # elif dataType =='RiverSkinDetection1':
    #     labelpath = skinAndWaterLabelPath
    #     # rgbpath = 'D:/ZY2006224YWJ/spectraldata/water_skin_rgb/'
    # else:
    #     labelpath = skinAndWaterLabelPath
    # rgbpath = 'D:/ZY2006224YWJ/spectraldata/water_skin_rgb/'
    dataFile_LEN = len(dataFile)
    for file_order, file in enumerate(dataFile):
        print(file)
        print(file_order, "/", dataFile_LEN)
        # output_log.writelines(str(file_order)+ "/" + str(dataFile_LEN))
        # 1 skin 2 cloth 3 other
        imgLabel = io.imread(labelpath + file + '.png')
        # t2 = time.time()
        # imgData = envi_loader(os.path.join(env_data_dir,file[:8])+'/', file,nora)
        # 读取img文件
        imgData = cv2.imread(rgbpath + file + '.png')
        imgData = imgData[:, :, ::-1]
        # t3 = time.time()
        # 没必要 特征变换 增加之前设计的斜率特征
        # H W 22
        # 11*11像素块 类别预测 作为中心像素类别 可以利用上下文信息 提升准确率
        # 分割出像素块 返回像素块和对应的类别
        if intervalSelect:
            print("intervalGenerateAroundDeltaPatchAllLen...")
            pix, label = intervalGenerateAroundDeltaPatchAllLen(imgData, imgLabel, num, length, file, 50, class_nums)
        else:
            print("random sampling...")
            pix, label = generateAroundDeltaPatchAllLen(imgData, imgLabel, num, length, class_nums)
        Data.extend(pix)
        Label.extend(label)
        del imgData
        gc.collect()
    # id = os.getpid()
    print("当前进程id: ", id, "done!!!!!!")
    return Data, Label


def multiProcessGenerateData(dataType=None, num=2500, length=11, bands=None, activate=None, nora=False, class_nums=4,
                             intervalSelect=True, featureTrans=False, rgbData=False, labelpath=None, imgpath=None,
                             addSpace=False):
    all_process_data = []
    all_process_label = []
    all_space_data = []
    if dataType == 'sea':
        dataFile = SeaFile
    # elif dataType == 'bmhWater':
    #     dataFile = bmhTrain
    elif dataType == 'RiverSkinDetection1':
        dataFile = RiverSkinDetection1
    elif dataType == 'RiverSkinDetection2':
        dataFile = RiverSkinDetection2
    elif dataType == 'RiverSkinDetection3':
        dataFile = RiverSkinDetection3
    elif dataType == 'RiverSkinDetectionAll':
        dataFile = RiverSkinDetection1 + RiverSkinDetection2 + RiverSkinDetection3
    elif dataType == 'draft':
        dataFile = hk_draft
    elif dataType == 'HZRiverSkinClothTrain':
        dataFile = HZRiverSkinClothTrain
    elif dataType == 'HZRiverSkinClothTest':
        dataFile = HZRiverSkinClothTest
    elif dataType == 'allTrain18':
        dataFile = allTrain18
    elif dataType == 'allTest18':
        dataFile = allTest18
    else:
        dataFile = RiverSkinDetectionTest
    # 启动多个进程 分发文件列表
    print(mp.cpu_count())
    process_nums = os.cpu_count()
    if process_nums > len(dataFile):
        process_nums = len(dataFile)
    id = os.getpid()
    # 获取当前进程id
    print("当前主进程id: ", id)
    # 进程池中从无到有创建process_nums个进程,以后一直是这process_nums个进程在执行任务 processes 参数默认值通过 os.cpu_count() 获取。
    process_poll = Pool(processes=process_nums)
    seg_data_all_process = []
    start = time.time()
    all_file_nums = len(dataFile)
    print("the len of dataFile is : ", all_file_nums)
    # output_log.writelines("the len of dataFile is : " + str(all_file_nums))
    perProcess = math.ceil(all_file_nums / process_nums)
    for i in range(process_nums):
        print(i)  # for循环会提前运行完毕，进程池内的任务还未执行。
        # 划分问价列表
        left = i * perProcess
        right = (i + 1) * perProcess if (i + 1) * perProcess < all_file_nums else all_file_nums
        # args : (dataType, dataFile, num, length, nora = True, class_nums = 2, intervalSelect = True, featureTrans = True)
        # 同步运行,阻塞、直到本次任务执行完毕拿到 single_res
        if rgbData:  # (dataFile, num, rgbpath, labelpath, length, class_nums = 2, intervalSelect = True)
            single_res = process_poll.apply_async(singleProcessGenerateRgbData, args=(
                dataFile[left:right], num, imgpath, labelpath, length, class_nums, intervalSelect,
            ))
        else:
            single_res = process_poll.apply_async(singleProcessGenerateData, args=(
                dataType, dataFile[left:right], num, length, bands, activate, nora, class_nums, intervalSelect,
                featureTrans, addSpace))
        # single_res = process_poll.apply_async(generateFun, args=(dataType, dataFile[left:right], num, length, bands, activate, nora, class_nums, intervalSelect, featureTrans))
        seg_data_all_process.append(single_res)  # 将调用apply_async方法，得到返回进程内存地址结果
    process_poll.close()
    process_poll.join()
    # gc.collect()
    for single_res in seg_data_all_process:
        # print('Single_process!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        tmpres = single_res.get()
        # print(len(tmpres[0]))
        # print(tmpres[0][0].shape)
        # print('Single_process_end!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        # tmpres 的 shape 可能不一致 所以
        all_process_data.extend(tmpres[0])
        all_process_label.extend(tmpres[1])
        if not rgbData and addSpace:
            all_space_data.extend(tmpres[2])
        del tmpres
        gc.collect()
    all_process_data = np.array(all_process_data, dtype=np.float32)
    all_process_label = np.array(all_process_label, dtype=np.int8)
    all_space_data = np.array(all_space_data, dtype=np.float32)
    stop = time.time()
    print('seg data cost time is %s' % (stop - start))
    # output_log.writelines('seg data cost time is %s' % (stop - start))
    # output_log.close()
    return all_process_data, all_process_label, all_space_data


def convert_to_one_hot(y, C):
    return np.eye(C, dtype=np.int8)[y.reshape(-1)]


# 随机选取 存在的问题 选取可能不均匀 可能边缘，或者一些 比较分歧、重要的地方不能保证选取到，数据多样性就不能保证
def generateAroundDeltaPatchAllLen(imgData, imgLabel, labelCount=2000, length=11, class_nums=2,
                                   overexposure_thre=45000, addSpace=False):
    row, col, d = imgData.shape
    imgData = np.array(imgData)
    # H W C -> C H W
    print("move overexposure area....")
    max_mask = np.max(imgData, axis=2)
    imgLabel[max_mask > overexposure_thre] = 0
    img = imgData.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    del max_mask
    gc.collect()
    # img.shape -> (1, channels, row, col)
    # 像素块变量 保存提取的像素块
    pix = np.empty([0, d, length, length], dtype=np.float32)
    pixSpace = np.empty([0, d, length, length], dtype=np.float32)
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
        per_class_num_care = [labelCount for _ in range(class_nums - 1)]
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
            # print(np.reshape(tmp, [-1, 4]), np.reshape(tmp, [-1, 4]).shape)
            labels = np.append(labels, np.reshape(tmp, [-1, class_nums]), axis=0)
            x = label_index[0][index]
            y = label_index[1][index]
            # B C H W
            if length % 2 == 0:
                pixAround = img[:, :, x - length1: x + length1, y - length1: y + length1]  # 3*3*4
            else:
                pixAround = img[:, :, x - length1: x + length1 + 1, y - length1: y + length1 + 1]  # 3*3*4
            # 堆叠像素块
            pix = np.append(pix, pixAround, axis=0)
    labels = np.array(labels).astype('int64')
    pix = np.array(pix, dtype=np.float32)
    return pix, labels, pixSpace


def intervalGenerateAroundDeltaPatchAllLen(imgData, imgLabel, labelCount=2000, length=11, filename='null', thre=500,
                                           class_nums=2, overexposure_thre=45000, addSpace=False):
    row, col, d = imgData.shape
    # img = np.expand_dims(imgData, 0)
    imgData = np.array(imgData)
    # H W C -> C H W
    # 去掉过曝区域
    print("move overexposure area....")
    max_mask = np.max(imgData, axis=2)
    imgLabel[max_mask > overexposure_thre] = 255  # 不参与训练
    img = imgData.transpose(2, 0, 1)
    # B C H W
    img = np.expand_dims(img, 0)  # b c h w

    img_spac = None
    if addSpace:
        img_spac = img.copy()  # B C H W
        img_spac_max = img_spac.copy()
        tup = (2, 3)
        # ori = img
        # 切面方向归一化
        for tdim in tup:
            img_spac_max = np.amax(img_spac_max, axis=tdim, keepdims=True)
        img_spac = img_spac / img_spac_max

    length1 = int(length / 2)
    pix = np.empty([0, d, length, length], dtype=np.float32)
    labels = np.empty([0, class_nums])
    pixSpace = np.empty([0, d, length, length], dtype=np.float32)

    for label in range(1, class_nums18 + 1):  # 0:other 1:skin 2:cloth 3:water 18通道皮肤衣物数据集是10分类标注的
        # cnt = 1
        label_index = np.where(imgLabel[length1:-length1, length1:-length1] == label)  # 找到标签值等于label的下标点 size：2*个数
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
        # 最小采样间隔
        interval = min_interval[label]
        # 间隔1采样，数量除以4
        if pixels_nums > interval * interval * labelCount:
            mul = math.sqrt(pixels_nums / labelCount)
            interval = round(mul)
        print('before :', filename, 'label:', label, 'interval ', interval, 'before :', pixels_nums)
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
            mask += [mk for mk in range(i * interval,
                                        interval * (i + 1) - 1 if interval * (i + 1) - 1 < imgLabel.shape[0] else
                                        imgLabel.shape[0])]
            # imgLabel_tmp[i*cnt:cnt*(i+1)-1,:]=0
        # print('mask',mask)
        imgLabel_tmp[mask, :] = 255
        new_label_index = np.where(imgLabel_tmp == label)
        print('after row sampling: ', len(new_label_index[0]))
        # print(imgLabel_tmp)
        # 隔列取样
        mask = []
        # print(filename, label, interval)
        # cnt = imgLabel.shape[1] // interval
        cnt = math.ceil(imgLabel.shape[1] / interval)
        # print('cnt: ', cnt)
        for i in range(cnt):
            mask += [mk for mk in range(i * interval,
                                        interval * (i + 1) - 1 if interval * (i + 1) - 1 < imgLabel.shape[1] else
                                        imgLabel.shape[1])]
        imgLabel_tmp[:, mask] = 255
        # print('mask', mask)
        # 取样后的索引
        new_label_index = np.where(imgLabel_tmp[length1:-length1, length1:-length1] == label)
        # 删掉边界像素！！

        new_count = len(new_label_index[0])
        print('after sampling colounm', filename, label, new_count)

        tmp = np.array(convert_to_one_hot(np.array([class_order for class_order in range(class_nums)]), class_nums)[
                           labelToclass18skin[label]])

        global class_nums_record
        class_nums_record[labelToclass18skin[label]] += new_count
        for index in range(new_count):
            labels = np.append(labels, np.reshape(tmp, [-1, class_nums]), axis=0)
            x = new_label_index[0][index]
            y = new_label_index[1][index]
            if length % 2 == 0:
                pixAround = img[:, :, x: x + 2 * length1, y: y + 2 * length1]  # 21,21,bands
                if addSpace:
                    pixSpaceAround = img_spac[:, :, x: x + 2 * length1, y: y + 2 * length1]
                else:
                    pixSpaceAround = np.empty([0, d, length, length], dtype=np.float32)
            else:
                pixAround = img[:, :, x: x + 2 * length1 + 1, y: y + 2 * length1 + 1]  # 21,21,bands
                if addSpace:
                    pixSpaceAround = img_spac[:, :, x: x + 2 * length1 + 1, y: y + 2 * length1 + 1]
                else:
                    pixSpaceAround = np.empty([0, d, length, length], dtype=np.float32)
            pix = np.append(pix, pixAround, axis=0)
            if addSpace:
                pixSpace = np.append(pixSpace, pixSpaceAround, axis=0)
            # np.save(save_path + filename + name_list[label - 1] + str(index).rjust(4, '0') + '.npy', pixAround)
    labels = np.array(labels).astype('int64')
    pix = np.array(pix, dtype=np.float32)
    pixSpace = np.array(pixSpace, dtype=np.float32)
    # print(pix.shape)
    # print('label shape:',labels.shape)
    # 这个不是 单张图 的抽样结果吧！！！
    print(filename, "sampling :", class_nums_record)
    # B C H W
    return pix, labels, pixSpace


def generateAroundDeltaPatchAllLen_6se(imgData, imgLabel, typeCode, labelCount=2000, length=11,
                                       selectMode=SELECT_RANDOM):
    row, col, d = imgData.shape
    imgData = np.array(imgData)
    img = imgData.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    # img.shape -> (1, channels, row, col)
    pix = np.empty([0, d, length, length], dtype=float)
    labels = np.empty([0, 4])
    length1 = int(length / 2)
    for label in range(1, 5):  # 共有14类，取值为1、2、3、...、14
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

        tmp = np.array(label2target[label % 4])

        if selectMode == SELECT_ALL:
            for index in range(pixels_nums):
                labels = np.append(labels, np.reshape(tmp, [-1, 4]), axis=0)
                x = label_index[0][index]
                y = label_index[1][index]
                pixAround = img[:, :, x - length1: x + length1 + 1, y - length1: y + length1 + 1]  # 3*3*4
                pix = np.append(pix, pixAround, axis=0)
        elif selectMode == SELECT_RANDOM:
            for num in range(labelCount):
                index = random.randint(0, pixels_nums - 1)  # 前闭后闭区间
                # labels = np.append(labels, label)
                # print(np.reshape(tmp, [-1, 4]), np.reshape(tmp, [-1, 4]).shape)
                labels = np.append(labels, np.reshape(tmp, [-1, 4]), axis=0)
                x = label_index[0][index]
                y = label_index[1][index]
                pixAround = img[:, :, x - length1: x + length1 + 1, y - length1: y + length1 + 1]  # 3*3*4
                pix = np.append(pix, pixAround, axis=0)

    labels = np.array(labels).astype('int64')
    pix = np.array(pix, dtype=float)
    # print(pix.shape)
    # print('label shape:',labels.shape)
    return pix, labels


if __name__ == '__main__':
    pass
