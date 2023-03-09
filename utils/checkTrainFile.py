import os
import shutil
import random
import numpy as np
from skimage import io
from data.dictNew import *
label_path = '/home/cjl/dataset_18ch/label/'
labellist = os.listdir(label_path)
labellist = [x[:-4] for x in labellist if x[-4:] == '.png']
png_path = '/home/cjl/dataset_18ch/raw_data/'
pnglist = os.listdir(png_path)
pnglist = [x for x in pnglist if x[-4:] == '.png']
skinClothRawPath = '/home/cjl/dataset_18ch/raw_data/'
waterRawPath = '/home/cjl/dataset_18ch/waterBmh/'

checklist = ['20211124150719_040_2.png','20211204154814.png', '20211124152449_080_2.png', '20211124152449_080_2.png']

# allSkinCloth18 = trainSkinCloth18 + testSkinCloth18
# allTest18 = testSkinCloth18 + testWater18

train_file = './all_train_file.txt'
test_file = './test_file.txt'

# trainf = open(train_file,'w')
# testf = open(test_file,'w')
#
# print('all train len : ', len(allSkinCloth18))
# test_list = random.sample(allSkinCloth18, int(len(allSkinCloth18)*0.3))
# for file in allSkinCloth18:
#     # if file[-4:] != '.png':
#     #     continue
#     if file in test_list:
#         testf.write('\'' + file + '\',' + '\n')
#     else:
#         trainf.write('\'' + file + '\',' + '\n')
#
# # os.close(trainf)
# # os.close(testf)
# trainf.close()
# testf.close()
print(len(allTrain18))
print(len(allTest18))
print(len(allTrain18) / len(allTest18))
print(len(trainWater18) / len(testWater18))
print(len(trainSkinCloth18) / len(testSkinCloth18))
for file in testSkinCloth18:
    if file in trainSkinCloth18:
        print(file)
# def change(a):
#     a[a==0] = 255
# for file in checklist:
#     imgLabel = io.imread(label_path + file)
#     change(imgLabel)
#     print(set(imgLabel.reshape(imgLabel.size)))
# 验证成功
# allTrainFile = trainSkinCloth18 + testSkinCloth18 + trainWater18 + testWater18
# print(len(allTrainFile))
# print(len(labellist))
# ten = 0
# three = 0
# tenClass_file = './tenClass_file.txt'
# threeClass_file = './threeClass_file.txt'
# tenClass_f = open(tenClass_file,'w')
# threeClass_f = open(threeClass_file,'w')
# for file in allTrainFile:
#     imgLabel = io.imread(label_path + file + '.png')
#     maxLabel = np.max(imgLabel)
#     if maxLabel == 10:
#         ten += 1
#         tenClass_f.write('\'' + file + '\',' + '\n')
#     if maxLabel == 3:  # 可能等于四！！4映射成0也是对的！！
#         three += 1
#         threeClass_f.write('\'' + file + '\',' + '\n')
# print(ten)
# print(three)
# print(ten + three)
# threeClass_f.close()
# tenClass_f.close()

# train_file = './trainData/all_train_file.txt'
# # test_file = './test_file.txt'
#
# trainf = open(train_file,'w')
#
# print('all train len : ', len(alltrainFileWater))
# for file in alltrainFileWater:
#     if file[-4:] == '.png':
#         # continue
#         trainf.write('\'' + file[:-4] + '\',' + '\n')
#     else:
#         trainf.write('\'' + file + '\',' + '\n')
# # os.close(trainf)
# # os.close(testf)
# trainf.close()
    # if file not in labellist:
    #     print(file, ' label not exist!!')
    # # if not os.path.exists(png_path + file + '.png'):
    # #     print(file, ' png not exist!!')
    # if not os.path.exists(skinClothRawPath + file + '.raw') or not os.path.exists(skinClothRawPath + file + '.hdr'):
    #     if not os.path.exists(waterRawPath + file + '.raw') or not os.path.exists(waterRawPath + file + '.hdr'):
    #         print(file, ' raw not exist!!')

# for

# for label in labellist

# for file in filelist:
#
#     if file[-4:]!='.png':
#         continue
#     imgLabel = io.imread(waterpath + file)
#     if np.max(imgLabel) == 2:
#         print(file)
#         shutil.copy(waterpath + file, shenzhenLabel + file)
    # print(np.max(imgLabel))
#     # print(os.path.getsize(imgpath + file[3:-4] + '.img') )
#     # break
#     if not os.path.exists(imgpath + file[3:-4] + '.img'):
#         print(file)
#         shutil.move(waterpath + file, imgpath + file)
#         continue
#
#     if not os.path.exists(imgpath + file[3:-4] + '.hdr'):
#         print(file)
#         shutil.move(waterpath + file, imgpath + file)
#         continue
#     if os.path.getsize(imgpath + file[3:-4] + '.img') != imgsize:
#         print(file, 'not equal imgsize!!')
#         shutil.move(waterpath + file, imgpath + file)
#         continue
#     # shutil.move(waterpath + file, labelpath + file)
#     waterFile.append(file[3:])
#     cnt += 1
# print(cnt)
# # 20210722
# samplewaterFile = random.sample(waterFile, 150)
# sampleSh = random.sample(trainFile, 350)
#
# shPngpath = '/home/cjl/dataset/rgb/'
# shLabelpath = '/home/cjl/dataset/label/'
# waterLabel = '/home/cjl/waterDataset/TO_WenJun/label/'
# randomselectShFile = ['20210714110639802', '20210714130847408']
# randomselectWaterFile = ['20220426172559745', '20220424162544512']
# dstpngpath = '/home/cjl/spectraldata/water_skin_rgb/'
# dstlabelpath = '/home/cjl/spectraldata/trainLabelAddWater/'

# for file in randomselectShFile:
#     shutil.copy(shPngpath + file + '.png', dstpngpath + file + '.png')
#     shutil.copy(shLabelpath + file + '.png', dstlabelpath + file + '.png')

# for file in randomselectWaterFile:
#     # shutil.copy(shPngpath + file + '.png', dstpngpath + file + '.png')
#     shutil.copy(waterLabel + 'rgb' + file + '.png', dstlabelpath + file + '.png')

# alltrainFileWater = shDataAndWater
# shData = trainFile
# waterFile = os.listdir(labelpath)
# waterFile = [x[3:-4] for x in waterFile]
# selectShFile = [x for x in shData if x not in alltrainFileWater]
# randomselectShFile = random.sample(selectShFile, 2)
# print(randomselectShFile)
# for file in randomselectShFile:
#     src = shpath + file[:8] + '/' + file + '.img'
#     dst = '/home/cjl/spectraldata/RiverLakeTrainData/'
#     dst = dst + file + '.img'
#     if not os.path.exists(src):
#         print(file, 'not exit')
#     else:
#         shutil.copy(src, dst)
#         src = src[:-4] + '.hdr'
#         dst = dst[:-4] + '.hdr'
#         shutil.copy(src, dst)
#
# selectWaterFile = [x for x in waterFile if x not in alltrainFileWater]
# randomselectWaterFile = random.sample(selectWaterFile, 2)
# print(randomselectWaterFile)
# for file in randomselectWaterFile:
#     src = imgpath + file + '.img'
#     dst = '/home/cjl/spectraldata/RiverLakeTrainData/'
#     dst = dst + file + '.img'
#     if not os.path.exists(src):
#         print(file, 'not exit')
#     else:
#         shutil.copy(src, dst)
#         src = src[:-4] + '.hdr'
#         dst = dst[:-4] + '.hdr'
#         shutil.copy(src, dst)
# 一个挑两张复制过去！

# for file in testFile:
#     if not os.path.exists(shpath + file[:8] + '/' + file + '.img') :
#         print(file)
#     if not os.path.exists(shpath + file[:8] + '/' + file + '.hdr') :
#         print(file)
#     if os.path.getsize(shpath + file[:8] + '/' + file + '.img') != imgsize:
#         print(file, 'not equal imgsize!!')
#         # shutil.move(waterpath + file, imgpath + file)
#         continue
# # 从trainFile中选350张
# for file in trainFile:
#     if not os.path.exists(shpath + file[:8] + '/' + file + '.img') :
#         print(file)
#     if not os.path.exists(shpath + file[:8] + '/' + file + '.hdr') :
#         print(file)
#     if os.path.getsize(shpath + file[:8] + '/' + file + '.img') != imgsize:
#         print(file, 'not equal imgsize!!')
#         # shutil.move(waterpath + file, imgpath + file)
#         continue
#
# train_file = './trainData/all_train_file.txt'
# # test_file = './test_file.txt'
#
# trainf = open(train_file,'w')
#
# print('all train len : ', len(alltrainFileWater))
# for file in alltrainFileWater:
#     if file[-4:] == '.png':
#         # continue
#         trainf.write('\'' + file[:-4] + '\',' + '\n')
#     else:
#         trainf.write('\'' + file + '\',' + '\n')
# # os.close(trainf)
# # os.close(testf)
# trainf.close()
