# from materialNet import *
import sys
sys.path.append('../')
from data.utilNetNew import *
from utils.load_spectral import *
from zw_cnn import MaterialModel,MaterialModel_leakrelu
from fc_cnn import FcCNN
from mac_CNN import MAC_CNN
import cv2
# import network
import gc
from utils.add_color import mask_color_img
from utils.os_helper import *
import csv
from utils.accuracy_helper import *
from sklearn.metrics import classification_report,f1_score
from utils.os_helper import mkdir
import math
from utils.parse_args import parse_test_args
import time
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'
# CUDA:0
args = parse_test_args()
# FOR_TESTSET =  args.FOR_TESTSET
# test_batch = args.test_batch
model_select = args.model_select
# print(FOR_TESTSET,test_batch)

bands = 9
CLASSES_NUM=4
device = torch.device('cuda')
mDevice=torch.device("cuda")

model_list = ['/home/cjl/ywj_code/code/rgb_model/fc_cnn_model_32_0.0001/35.pkl','/home/cjl/ywj_code/code/rgb_model/mac_cnn_model_32_0.0001/35.pkl']
png = '20210329145952293'
envi_path = '/home/cjl/data/envi/'
label_path = '/home/cjl/data/label/'
png_path = '/home/cjl/data/png/'

color_class = [[0,0,255],[255,0,0],[0,255,0]]
if model_select==0:
    model = FcCNN(4).cuda()
else:
    model = MAC_CNN(4).cuda()
model_path = model_list[model_select]
model.load_state_dict(torch.load(model_path))
model.eval()
t1 = time.time()
# imgLabel = io.imread(label_data_dir + png + '.png')
        # t2 = time.time()
# imgData,t2 = ssrn_envi_loader_cut_scale(env_data_dir, png)
imgData = envi_rgbnir_loader(envi_path, png)
# print("load data cost time: ",t2-t1)
# t3 = time.time()
# print("envi_normalize cost time: ",t3-t2)

with torch.no_grad():
    # imgData = imgData.permute(0, 2, 3, 1)
    imgData = torch.from_numpy(imgData)
    imgData = Variable(imgData).float().to(device)
    imgData = imgData.permute(2, 0, 1)
    # imgData = Variable(imgData).float().to(device)
    imgData = imgData.unsqueeze(0)
    imgData = imgData[:,:3]
    # imgData = imgData.unsqueeze(0)#B C H W D
    # imgData = imgData.unsqueeze(0)
    # inputData = imgData.permute(0,3,1,2)
    #切块变成批量测试
    t4 = time.time()
    if model_select==0:
        pix_length = 32
    else:
        pix_length = 16
    pix_data = torch.empty([0, 3, pix_length, pix_length], dtype=torch.float).to(device)
    for x in range(pix_length//2, imgData.size()[2] - pix_length//2):
        # for y in range(5, imgData.size()[3] - 5):
        for y in range(pix_length//2, pix_length//2+1):
            pix_data = torch.cat((pix_data, imgData[:, :, x - pix_length//2:x + pix_length//2, y - pix_length//2:y + pix_length//2]), 0)
    predict = model(pix_data)
    # predict = torch.squeeze(predict)
    predict_ind = torch.argmax(predict, dim=1) # B：只有一个维度
    predict_ind = predict_ind.reshape(1,imgData.size()[2]-pix_length)
    t5 = time.time()
    print("predict 1849 pixels cost time: ",t5-t4)
    if model_select==0:
        print("predict 1827 * 1383 pixels cost time :", 1383*(t5-t4))
    else:
        print("predict 1843 * 1399 pixels cost time :", 1399 * (t5 - t4))
    # predict_ind = predict_ind.cpu().detach().numpy()
    # #print(np.unique(predict_ind))
    # imgGt = cv2.imread(png_path + png + '.png')
    # imgRes=imgGt
    # for color_num in [1,2,3]:
    #     imgRes = mask_color_img(imgRes,mask=(predict_ind == color_num),color=color_class[color_num-1],alpha=0.7 )
    # cv2.imwrite(png + '_'+str(model_select)+'_pre.jpg', imgRes)
    #
    # img_label = cv2.imread(label_data_dir + png + '.png', cv2.IMREAD_GRAYSCALE)
    # # img_label = img_label[5:5+rows, 5:5+cols]
    # img_label = img_label[5:-5,5:-5]
    # img_label = transformLabel(img_label, MUTI_CLASS)
    # for color_num in [1,2,3]:
    #     imgGt = mask_color_img(imgGt,mask=(img_label == color_num),color=color_class[color_num-1],alpha=0.7 )
    # cv2.imwrite(png + '_'+str(model_select)+'_gt.jpg', imgGt)







