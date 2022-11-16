from data.dictNew import *
import os

bmh_label = '/home/cjl/dataset_18ch/WaterLabel_mask_221011/'
bmh_png_path = '/home/cjl/dataset_18ch/waterBmh/'
bmh_raw_path = '/home/cjl/dataset_18ch/waterBmh/'

for file in bmhTrain:
    if not os.path.exists(bmh_label + file + '.png'):
        print("label:",file)
    if not os.path.exists(bmh_png_path + file + '.png'):
        print("png:", file)
    if not os.path.exists(bmh_raw_path + file + '.raw'):
        print("raw:", file)
