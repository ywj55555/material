import os
import re
# from utils.os_helper import mkdir
import shutil

labelpath = 'D:/ZY2006224YWJ/spectraldata/label/.imageLabelingSessionwater2_SessionData/'
pngpath = 'D:/ZY2006224YWJ/spectraldata/RiverSkinDetection2/'
labels = os.listdir(labelpath)
labels.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
pngs = os.listdir(pngpath)

for label, png in zip(labels, pngs):
    os.rename(labelpath+label, labelpath+png)



# filepath = 'D:/dataset/lg/Label/'
# det_path = 'D:/dataset/lg/needmark/'
#
# det_name = os.listdir(det_path)
# det_name2 = [ png for png in det_name if png[-4:]=='.png']
# filelist = os.listdir(filepath)
# filelist.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))  # 返回字符串里的数字
#
# cnt=0
# for file in filelist:
#     if file[-4:]!='.png':
#         continue
#     print(file)
#
#     os.rename(filepath + '\\' + file, filepath + '\\' + det_name2[cnt])
#     cnt = cnt + 1
# filepath2 = r'D:\ZY2006224YWJ\material-extraction\6sensor_data\102\.6sensor_SessionData'
#     # filepath2 = filepath+file
# filelist2 = os.listdir(filepath2)
#     # filelist2.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))  # 返回字符串里的数字
#     for png in filelist2:
#         # if file[:6]!='.image':
#         #     continue
#         print(png)
#         # dir = filepath2+'\\'+file
#         # filelist3 = os.listdir(dir)
#         # filelist3.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))  # 返回字符串里的数字
#         # for png in filelist3:
#         os.rename(filepath2+'\\'+png,filepath2+'\\'+det_name[cnt])
#         print(cnt)
#         cnt+=1
        # print(file)

    # filelist3 = os.listdir(dir)
    # filelist2.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))  # 返回字符串里的数字
    # for png in filelist2:
    #     os.rename(filepath2+'\\'+png,filepath2+'\\'+filelist[cnt])
    #     print(cnt)
    #     cnt+=1