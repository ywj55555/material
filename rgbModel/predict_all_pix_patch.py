'''
每张图生成2250个测试像素，间隔取样2000个点左右（标注的点）展开成一维取样？？二维间隔取样才能保证稀疏性，更加模拟整图预测
网格取样需要每一类单独计算间隔，然后按照每一类别数量在整图的比例进行取样
'''
import sys
sys.path.append('../')
from tqdm import tqdm
import csv
import torch
from model_block import network
from utils.load_spectral import envi_loader
from utils.accuracy_helper import *
from utils.os_helper import mkdir
# from sklearn.metrics import classification_report
from utils.load_spectral import kindsOfFeatureTransformation,envi_normalize
from utils.parse_args import parse_test_args
import os
import math
import time
from model_block.DBDA_Conv import DBDA_network_MISH_full_conv
from utils.add_color import mask_color_img
from zw_cnn import MaterialModel,MaterialModel_leakrelu
from fc_cnn import FcCNN
from mac_CNN import MAC_CNN
from model1 import Model1

waterLabelPath = 'D:/dataset/lg/Label_rename/'
waterImgRootPath = 'D:/dataset/lgimg/train/'
png_path = '/home/cjl/ssd/dataset/water_skin_rgb/'
label_path = '/home/cjl/ssd/dataset/water_skin_label/'

log = './log/'
mkdir(log)
det_num = 15000
train_mean = np.array([56.54884781, 45.74997012, 39.97695533])
# train_mean = np.array(train_mean)
train_var = np.array([3327.70049172, 1916.7318851, 1372.06279076])
train_mean = torch.from_numpy(train_mean).float().cuda()
train_var = torch.from_numpy(train_var).float().cuda()
# train_var = np.array(train_var)
color_class = [[0,0,255],[255,0,0],[0,255,0],[255,0,255]]

args = parse_test_args()
model_select = args.model_select
print('model select: ',model_select)
model_str = {0:'_FcCNN_',1:'_SSRN_'}
length_list = {0:11,1:32,2:16,3:11,4:32}
epoch = str(10)
model_list = ['./fc_cnn32_0.0001/5.pkl',
              './fc_cnn32_0.0001/24.pkl']

class_nums = 4
device = torch.device('cuda')
if model_select == 0:
    model = MaterialModel().to(device)
elif model_select == 1:
    model = FcCNN(class_nums, False).to(device)
elif model_select == 2:
    model = MAC_CNN(class_nums).to(device)
elif model_select == 3:
    model = MaterialModel_leakrelu().to(device)
else:
    model = Model1(4).to(device)
    # model = network.SSRN_network(bands, CLASSES_NUM).to(device)
model_path = model_list[model_select - 1]
model.load_state_dict(torch.load(model_path))
model.eval()

result_dir = './res/'+model_str[model_select]+'pre_label'+'/' + epoch + '/'
result_pred = './resTrain/' + "fc_cnn/" + epoch + '/'
mkdir(result_dir)
mkdir(result_pred)
output_log = open('./log/output_test_DBDA.log','w')

label_list = []
predict_list = []
t1 = time.time()
# tmpfile = [x for x in alltrainFile if x not in dfx_test]
# file_list = RiverSkinDetection1 + RiverSkinDetection2 + RiverSkinDetection3
file_list = RiverSkinDetectionAllTest
# file_list = waterFile

# file_list = [x[3:] for x in file_list]
print("the number of test file:",len(file_list))
with torch.no_grad():
    for file in file_list:
        begin = time.time()

        imgData = cv2.imread(png_path + file + '.png')
        imgData = imgData[:, :, ::-1].copy()

        tmp_w = imgData.shape[0] - (length_list[model_select]//2) * 2
        tmp_h = imgData.shape[1] - (length_list[model_select]//2) * 2
        label_coor = [[x,y] for x in range(tmp_w) for y in range(tmp_h)]
        label_coor = np.array(label_coor)

        pred_total = 0
        pred_correct = 0
        imgData = torch.from_numpy(imgData).float().cuda()

        imgData = imgData.unsqueeze(0)  # B H W C

        cnt = math.ceil(len(label_coor)/det_num)
        predict_png = []
        for i in tqdm(range(cnt)):
            coor_tmp = label_coor[i * det_num:(i + 1) * det_num if (i + 1) < cnt else len(label_coor)]
            # imgData = []
            pix_data = torch.empty([0, length_list[model_select], length_list[model_select], 3], dtype=torch.float).cuda()
            for coor in coor_tmp:
                pix_data = torch.cat((pix_data, imgData[:, coor[0] :coor[0]+ length_list[model_select],
                                                coor[1]:coor[1] + length_list[model_select], :]), 0)
            # pix_data = pix_data.permute(0, 2, 3, 1) # b h w c
        #     切面归一化：
            pix_data_ = pix_data.view(np.prod(pix_data.shape[:3]),pix_data.shape[3])

            pix_data_ = (pix_data_-train_mean)/torch.sqrt(train_var)
            pix_data = pix_data_.view(pix_data.shape)
            pix_data = pix_data.permute(0, 3, 1, 2)

            predict = model(pix_data)

        # print(predict.shape[0])
        # predict = torch.squeeze(predict)
            predict_ind = torch.argmax(predict, dim=1)  # B：只有一个维度
            del predict
            del pix_data
            predict_ind = predict_ind.squeeze() #B
            predict_ind = predict_ind.cpu().numpy()
            predict_list.extend(predict_ind)
            predict_png.extend(predict_ind)
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
        predict_png = np.array(predict_png)
        predict_png = predict_png.reshape(tmp_w,tmp_h)
        print(predict_png.shape)
        imgGt = cv2.imread(png_path + file + '.png')
        imgGt = imgGt[length_list[model_select]//2:-length_list[model_select]//2,
                    length_list[model_select]//2:-length_list[model_select]//2]
        # imgGt2 = imgGt.copy()
        print(imgGt.shape)
        for color_num in range(1, class_nums):
            imgGt = mask_color_img(imgGt,mask=(predict_png == color_num),color=color_class[color_num-1],alpha=0.7 )
        cv2.imwrite(result_pred + file + '.png', imgGt)
        cv2.imwrite(result_dir + file + '_preLabel.png', predict_png)
        end = time.time()
        print('cost time : ', end - begin)
        del imgData
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()

t2 = time.time()

print("cost time",t2-t1,"s")
print("average cost time",(t2-t1) / len(file_list),"s")









