import os
import sys
sys.path.append('../')
from data.dictNew import *
import random

# all_list = trainFile
# test_list = random.sample(all_list,int(len(all_list)*0.2))

'''
lable_path = r'E:/raw_file/getImage_data_1204/'
filelist = os.listdir(lable_path)
print(len(filelist))
tmp_file = './train_add_file.txt'
trainf = open(tmp_file,'w')
for i in range(len(filelist)):
    if filelist[i][-4:]!='.png':
        continue
    trainf.write('\'' + filelist[i][:-4] + '\',' + '\n')
trainf.close()
'''

tainpath = 'D:/dataset/shuichi_label/'
trainFile = os.listdir(tainpath)
# test_list = random.sample(trainFile,int(len(trainFile)*0.2))

train_file = './all_train_file.txt'
# test_file = './test_file.txt'

trainf = open(train_file,'w')
# testf = open(test_file,'w')
# trainfile = RiverSkinDetection1 + RiverSkinDetection2 + RiverSkinDetection3

for file in trainFile:
    if file[-4:] != '.png':
        continue
    # if file in test_list:
    #     testf.write('\''+file[:-4]+'\','+'\n')
    # else:
    trainf.write('\'' + file[:-4] + '\',' + '\n')

# os.close(trainf)
# os.close(testf)
trainf.close()
# testf.close()






