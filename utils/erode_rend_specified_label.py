# mask_color_img(img, mask, color=[255, 0, 255], alpha=0.7)
# from erode_label import open_demo
import cv2

from utils.add_color import mask_color_img
from utils.paint_rect import dilate_specific_label, erode_all_label,open_specific_label_difsize

# png_path = r'E:\raw_file\1019data\hidden-object'+'\\'
# save_path = r'E:\raw_file\predict_result\erode_hidden_object2'+'\\'
# mkdir(save_path)
# label_path = r'E:\raw_file\predict_result\predict_label'+'\\'
# png_name = ['20211019165650','20211019161026'] #short - long shirt
# png_name = ['20211019170656','20211019170709','20211019170721'] #logo
# png_name = [
# '20211019165650',
# '20211019165659',
# '20211019165705',
# '20211019165711',
# '20211019165718',
# '20211019165723',
# '20211019165729',
# '20211019165736',
# '20211019165743',
# '20211019165749',
# '20211019165755',
# '20211019165801',
# '20211019165806',
# '20211019165812',
# '20211019165818',
# '20211019165823',
# '20211019165828'
# ]#zhuanti
# png_name = ['20211022143336','20211022143105']#zhike mask
# png_name = ['20211021142145','20211021143545'] #xiangjiao mask
# png_name =[
# '20211019162357',
# '20211019162407',
# '20211019162414',
# '20211019162424',
# '20211019162434',
# '20211019162443',
# '20211019162454',
# '20211019162501',
# '20211019162508',
# '20211019162513',
# '20211019162521',
# '20211019162529'
#
# ]#hidden-object

# png_name = [
# '20211022150249',
# '20211022150319',
# '20211022150323',
# '20211022150328',
# '20211022150332',
# '20211022150350',
# '20211022150356',
# '20211022150400',
# '20211022150405',
# '20211022150412',
# '20211022150418',
# '20211022150428',
# '20211022150436',
# '20211022150447',
# '20211022150454',
# '20211022150512',
# '20211022150518',
# '20211022150523',
# '20211022150531',
# '20211022150539',
# '20211022150547',
# '20211022150553',
# '20211022150601',
# '20211022150615',
# '20211022150649',
# '20211022150654',
# '20211022150700',
# '20211022150706',
# '20211022150712',
# '20211022150718'
#
# ]

specified_label_list = [3, 2, 1]
color_list = [[0,0,255],[255,0,0], [0,255,0]]

res_path = './res/'
# mkdir(res_path)
label_path = '/home/cjl/ywj_code/graduationCode/alien_material/result/twoBranch3.0_skinClothWater18/extraTest_label/6/'
all_png_path = '/home/cjl/dataset_18ch/raw_data/'
# label_name = '20220623124427876.png'
# label_name = '20220623124419718.png'
# label_name = '20220623143139009.png'
label_name ='20221120110439.png'
png_data = cv2.imread(all_png_path+label_name)
png_data2 = png_data.copy()
# png_data = png_data[5:-5, 5:-5]
# png_data = np.zeros_like(png_data)
label_data = cv2.imread(label_path + label_name, cv2.IMREAD_GRAYSCALE)
# label_data_tmp = np.zeros_like(label_data)
# label_data_tmp[label_data == 1] = 1
# label_data_tmp[label_data == 2] = 2
# label_data_tmp[label_data == 3] = 3
# 先膨胀 dilate_specific_label(dst, dilate_size, label)
dilate_size = [8, 12, 10]
for i, specified_label in enumerate(specified_label_list):  # skin cloth water  reverse
    label_data = dilate_specific_label(label_data, dilate_size[i], specified_label)

for specified_label in specified_label_list:
    png_data2 = mask_color_img(png_data2,label_data==specified_label,color=[color_list[specified_label-1]])
cv2.imwrite(res_path + label_name[:-4] + '_dilate.png',png_data2)
# 然后整体开运算
dilate_size = 10
label_data = erode_all_label(label_data, dilate_size)
# 逐类别腐蚀
# for specified_label in specified_label_list:  # skin cloth water
#     label_data = open_specific_label_difsize(label_data,8,15,specified_label)

# label_data = open_demo(label_data_tmp,15)
# res_png = png_data
label_data = open_specific_label_difsize(label_data,8,15,3)
for specified_label in specified_label_list:
    png_data = mask_color_img(png_data,label_data==specified_label,color=[color_list[specified_label-1]])
cv2.imwrite(res_path + label_name,png_data)

# for png in png_name:
#     print(png)
#     png_data = cv2.imread(png_path+png+'.png')
#     # png_data = np.zeros_like(png_data)
#     label_data = cv2.imread(label_path+png+'.png',cv2.IMREAD_GRAYSCALE)
#     label_data_tmp = np.zeros_like(label_data)
#     label_data_tmp[label_data==1]=1
#     label_data_tmp[label_data == 2] = 2
#     for specified_label in specified_label_list:
#         label_data = open_specific_label(label_data_tmp,15,specified_label)
#
#     # label_data = open_demo(label_data_tmp,15)
#     # res_png = png_data
#     for specified_label in specified_label_list:
#         png_data = mask_color_img(png_data,label_data==specified_label,color=[color_list[specified_label-1]])
#     cv2.imwrite(save_path+png+'.png',png_data)
