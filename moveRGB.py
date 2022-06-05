import shutil
import os
from data.dictNew import *
from utils.os_helper import mkdir

rootpath = 'E:/tmp/lg/'
waterPath = 'E:/tmp/hzWater/'
mkdir(waterPath)
moveFile = waterFile
pathlist = os.listdir(rootpath)
pathlist = [x for x in pathlist if x[-3:] == 'rgb']
cnt = 0
for png in moveFile:
    for tmp_path in pathlist:
        oripath = rootpath + tmp_path + '/' + png + '.tif'
        dstpath = waterPath + png + '.png'
        if os.path.exists(oripath):
            shutil.copy(oripath, dstpath)
            cnt+=1
            break
print(cnt)
