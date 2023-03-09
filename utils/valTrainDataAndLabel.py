from data.dictNew import *
import os

# bmh_label = '/home/cjl/dataset_18ch/WaterLabel_mask_221011/'
# bmh_png_path = '/home/cjl/dataset_18ch/waterBmh/'
# bmh_raw_path = '/home/cjl/dataset_18ch/waterBmh/'
#
# for file in bmhTrain:
#     if not os.path.exists(bmh_label + file + '.png'):
#         print("label:",file)
#     if not os.path.exists(bmh_png_path + file + '.png'):
#         print("png:", file)
#     if not os.path.exists(bmh_raw_path + file + '.raw'):
#         print("raw:", file)

labelpath = '/home/cjl/spectraldata/trainLabelAddWater/'

# # rgbpath = '/home/cjl/spectraldata/water_skin_rgb/'
# imgpath = '/home/cjl/spectraldata/RiverLakeTrainData/'
# filelist = os.listdir(labelpath)
# cnt = 0
# for file in filelist:
#     if file[-4:] != '.png':
#         continue
#     if not os.path.exists(imgpath + file[:-4] + '.hdr'):
#         print(file, 'not exit hdr')
#     elif not os.path.exists(imgpath + file[:-4] + '.img'):
#         print(file, 'not exit img')
#     else:
#         cnt += 1
# print(cnt)