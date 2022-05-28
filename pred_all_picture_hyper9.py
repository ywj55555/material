'''
每张图生成2250个测试像素，间隔取样2000个点左右（标注的点）展开成一维取样？？二维间隔取样才能保证稀疏性，更加模拟整图预测
网格取样需要每一类单独计算间隔，然后按照每一类别数量在整图的比例进行取样
'''
from data.dictNew import testFile
import numpy as np
import cv2
import csv
import torch
import network
import spectral.io.envi as envi
from utils.load_spectral import envi_loader
from utils.accuracy_helper import *
from utils.os_helper import mkdir
from sklearn.metrics import classification_report
from utils.parse_args import parse_test_args
import os
import math
import time
from sklearn import preprocessing
from skimage import io
from utils.add_color import mask_color_img


envi_dir = '/home/cjl/dataset/hyperSpectral/'
label_dir = '/home/cjl/dataset/hyperSpectral/'
png_dir = '/home/cjl/dataset/hyperSpectralPng/'
png_path =  '/home/cjl/dataset/hyperSpectralPng/'

log = './log/'
mkdir(log)
det_num = 15000
# train_mean = np.array([1638.30293365, 1856.07645628, 2395.36102414, 2963.61267338, 3825.72213097,
#  3501.42091296, 4251.15954562, 3965.7991342,  3312.6616123 ])
# # train_mean = np.array(train_mean)
# train_var = np.array([ 5402022.45076405,  6679240.29294402,  9629853.43114098, 15170998.1931259,
#  25767233.41870246, 18716092.38116236, 18231725.96313715, 28552394.38170321,
#  13127436.03417886])
# train_mean = torch.from_numpy(train_mean).float().cuda()
# train_var = torch.from_numpy(train_var).float().cuda()
# train_var = np.array(train_var)

args = parse_test_args()
# model_select = args.model_select
data_type = args.data_type
# print('model select: ',model_select)
print('data_type',data_type)
model_str = {0:'_DBDA_',1:'_SSRN_'}
model_list = ['/home/cjl/ywj_code/contrast_model_sh/DBDA_model_hyper9meat_meat_bands_32_0.0005_5/392.pkl',
              '/home/cjl/ywj_code/contrast_model_sh/DBDA_model_hyper9oil_32_0.0005_3/49.pkl',
              '/home/cjl/ywj_code/contrast_model_sh/DBDA_model_hyper9fruit_32_0.0005_6/49.pkl',
              '/home/cjl/ywj_code/contrast_model_sh/DBDA_model_hyper9fruit8_32_0.0005_8/41.pkl']
bands = 9
# select_bands = [28,56,75,83,108,125,133,143,173]
# meat_bands = [15,50,75,104,146,157,177,209,241]
if data_type == 'meat':
    dataFile = meatFile
    CLASSES_NUM = 5
    model_select = 0
    color_class = [[0, 0, 255], [255, 0, 0], [0, 255, 0],[	255	,153,18]]
    select_bands = [15,50,75,104,146,157,177,209,241]
elif data_type == 'fruit':
    dataFile = fruitFile
    CLASSES_NUM = 6
    model_select = 2
    color_class = [[227,	207,	87],[	0,	255	,255], [0, 255, 0],[0, 0, 255],[255,	97,	0]]
    select_bands = [28, 56, 75, 83, 108, 125, 133, 143, 173]
elif data_type == 'fruit8':
    dataFile = fruitFile
    CLASSES_NUM = 8
    model_select = 3
    # 颜色重复了，操蛋
    color_class = [[227,	207,	87],[	0,	255	,255], [0, 255, 0],[0, 0, 255],[255,	97,	0],[255,97,3],[	189,252,201]]
    select_bands = [28, 56, 75, 83, 108, 125, 133, 143, 173]
elif data_type == 'oil':
    dataFile = oilFile
    CLASSES_NUM = 3
    model_select = 1
    color_class = [[0, 0, 255], [255, 0, 0]]
    select_bands = [28, 56, 75, 83, 108, 125, 133, 143, 173]
else:
    dataFile = testfile
    CLASSES_NUM = 5
    model_select = 0
    color_class = [[0, 0, 255], [255, 0, 0], [0, 255, 0]]
    select_bands = [28, 56, 75, 83, 108, 125, 133, 143, 173]
device = torch.device('cuda')



length = 11
model = network.DBDA_network_MISH(bands, CLASSES_NUM).to(device)

model_path = model_list[model_select]
model.load_state_dict(torch.load(model_path))
model.eval()

csv2_save_path = log+'pix_predict'+data_type+'9bands.csv'
f2 = open(csv2_save_path, 'w', newline='')
f2_csv = csv.writer(f2)
# csv2_header = ['micro accuracy','three_acc']
csv2_header = ['micro accuracy']
f2_csv.writerow(csv2_header)


result_dir = './res/'+data_type+'_9bands_pre_png'+'/'
mkdir(result_dir)

label_list = []
predict_list = []
t1 = time.time()
with torch.no_grad():
    for file in dataFile:
        # imgLabel = io.imread(label_dir + file[:-4] + '.png')
        if data_type=='fruit8':
            imgLabel = io.imread(label_dir + file[:-4] + '_8.png')
        else:
            imgLabel = io.imread(label_dir + file[:-4] + '.png')
        # 稀疏采样 这样会类别不均 最好在 label循环里面进行 回到稀疏采样代码
        # alldataNums = np.sum(imgLabel[length1:-length1, length1:-length1] !=0)
        imgLabel[imgLabel == 0] = 255
        imgLabel[imgLabel == CLASSES_NUM] = 0  # 转换背景
        # print('the number of datapath: ',alldataNums)
        enviData = envi.open(envi_dir + file[:-4] + '.hdr', envi_dir + file)
        enviData = enviData.load()
        enviData = np.array(enviData, dtype=float)
        imgData = enviData.reshape(np.prod(enviData.shape[:2]), np.prod(enviData.shape[2:]))
        imgData = preprocessing.scale(imgData)
        traindata = imgData.reshape(enviData.shape[0], enviData.shape[1], enviData.shape[2])
        traindata = np.expand_dims(traindata, 0)  # B H W C
        traindata = traindata[:,:,:,select_bands]

        label_tmp = imgLabel[5:-5,5:-5]
        # label_tmp = transformLabel(label_tmp, MUTI_CLASS_SH) #255,0,1,2,3
        label_fla = label_tmp.flatten()
        tmp_w = label_tmp.shape[0]
        tmp_h = label_tmp.shape[1]
        label_coor = [[x,y] for x in range(tmp_w) for y in range(tmp_h)]
        label_coor = np.array(label_coor)
        # remain_index = label_fla!=255
        # remain_pix = label_fla[remain_index]

        # remain_coor = label_coor[remain_index]

        # interval = remain_pix.shape[0]//det_num
        # # det_pix = remain_pix[::interval]
        # det_coor = remain_coor[::interval]
        # remain_pix = remain_pix[::interval]
        label_list.extend(label_fla)
        # print(det_coor.shape[0])
        # img_input = np.empty()
        pred_total = 0
        pred_correct = 0
        imgData = torch.from_numpy(traindata).float().cuda()
        labelData = torch.from_numpy(label_fla).long().cuda()


        # pred_total+=labelData.size()[0]
        # imgdata = imgdata.float().cuda()
        imgData = imgData.unsqueeze(1)  # B C H W D
        # imgData = imgData.unsqueeze(0)

        cnt = math.ceil(len(label_coor)/det_num)
        predict_png = []
        for i in range(cnt):
            coor_tmp = label_coor[i * det_num:(i + 1) * det_num if (i + 1) < cnt else len(label_coor)]
            # imgData = []
            pix_data = torch.empty([0, 1, length, length, bands], dtype=torch.float).cuda()
            for coor in coor_tmp:
                pix_data = torch.cat((pix_data, imgData[:, :, coor[0] :coor[0]+ length, coor[1]:coor[1] +length, :]), 0)
        # pix_data_ = pix_data.permute()
        #     pix_data_ = pix_data.view(np.prod(pix_data.shape[:4]),pix_data.shape[4])

            # pix_data_ = (pix_data_-train_mean)/torch.sqrt(train_var)

            # pix_data = pix_data_.view(pix_data.shape)
            predict = model(pix_data)
        # print(predict.shape[0])
        # predict = torch.squeeze(predict)
            predict_ind = torch.argmax(predict, dim=1)  # B：只有一个维度
            predict_ind = predict_ind.squeeze() #B
            pred_correct += (predict_ind == labelData[i * det_num:(i + 1) * det_num if (i + 1) < cnt else len(label_coor)]).sum().item()

            predict_ind = predict_ind.cpu().numpy()
            predict_list.extend(predict_ind)
            predict_png.extend(predict_ind)
        predict_png = np.array(predict_png)
        predict_png = predict_png.reshape(imgData.size()[2]-10,imgData.size()[3]-10)

        imgGt = cv2.imread(png_path + file[:-4] + '.png')
        imgGt = imgGt[5:-5,5:-5]
        # imgGt2 = imgGt.copy()
        for color_num in range(1,CLASSES_NUM):
            imgGt = mask_color_img(imgGt,mask=(predict_png == color_num),color=color_class[color_num-1],alpha=0.7 )

        cv2.imwrite(result_dir + file[:-4] + '49.png', imgGt)
        cv2.imwrite(result_dir + file[:-4] + '_label49.png', predict_png)

        # for color_num in [1, 2,3]:
        #     imgGt2 = mask_color_img(imgGt2,mask=(label_tmp == color_num),color=color_class[color_num-1],alpha=0.7 )
        # imgGt2 = mask_color_img(imgGt2, mask=label_tmp == 0 , color=[125,125,125],
        #                       alpha=0.7)
        # cv2.imwrite(result_dir + file + '_gt.png', imgGt2)


# label_list[label_list=]
label_list = np.array(label_list).flatten()
predict_list = np.array(predict_list).flatten()
remain_pix = label_list!=255
label_list = label_list[remain_pix]
predict_list = predict_list[remain_pix]

acc_nums = np.sum(label_list==predict_list)
t2 = time.time()
all_acc =acc_nums/len(label_list)



print("all acc:",all_acc)


# iou = np.sum(np.logical_and(label_list==predict_list,label_list!=0))
# all_cnts = np.sum(label_list==1)+np.sum(label_list==2)+np.sum(label_list==3)
# three_acc = iou/all_cnts
# print("three acc:",three_acc)

print("cost time",t2-t1,"s")

if data_type == "meat":
    target_names = ['other', 'beef', 'mutton', 'chicken', 'duck']
elif data_type == 'fruit':
    target_names = ['other', 'banana', 'green_pear', 'green_vegetables', 'passion_fruit', 'tangerinr']
elif data_type == 'fruit8':
    target_names = ['other', 'banana','green_pear','green_vegetables','passion_fruit','Gong_orange','wokan','apple']
elif data_type == 'oil':
    target_names = ['other', 'oil', 'sand']
else:
    target_names = ['other', 'beef', 'mutton', 'chicken', 'duck']

res = classification_report(label_list, predict_list, target_names=target_names,output_dict=True)
#
csv2_line = [all_acc]
for k in target_names:
    # print(k,'skin pre=', res['skin']['precision'], 'skin rec=', res['skin']['recall'])
    print(k, ' pre=', res[k]['precision'], ' rec=', res[k]['recall'], ' f1-score=', res[k]['f1-score'])
    csv2_line.extend([res[k]['precision'], res[k]['recall'], res[k]['f1-score']])
print('all test micro accuracy:', res['accuracy'], ' all test macro avg f1:', res['macro avg']['f1-score'])
f2_csv.writerow(csv2_line)
f2.close()









