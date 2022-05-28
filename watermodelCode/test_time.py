from materialNet import *
from data.utilNetNew import *
from utils.load_spectral import *
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

model_list = ['/home/cjl/ywj_code/code/ori_multi-category/ori_model_add_sh/19.pkl','/home/cjl/ywj_code/code/Multi-category_all/model_focal_lr5-4_weight2-exp3_add_sh/100.pkl']
png_list =['20210329154653734',
    '20210329154725971',#读取出错
#    '20210329154859783',
    '20210329155450683',
    '20210329155532031',
    # shanghai_data:
    '20210521101823861',
    '20210521102712957',
    '20210521102635825',
    '20210521095803026',
    '20210521102422271',
    '20210521124514556']
envi_path = '/home/cjl/data/envi/'
label_path = '/home/cjl/data/label/'
png_path = '/home/cjl/data/png/'

# color_class = [[0,0,255],[255,0,0],[0,255,0]]
model = MaterialSubModel(20, 4).cuda()
model_path = model_list[model_select]
model.load_state_dict(torch.load(model_path))
model.eval()
envi_normalize_list = []
predict_list = []

for png in png_list:

    imgData ,t1 = envi_loader(envi_path, png)
    imgData = transform2(imgData)
    t2 = time.time()
    # imgData,t2 = ssrn_envi_loader_cut_scale(env_data_dir, png)
    # print("load data cost time: ",t2-t1)
    # t3 = time.time()
    print("envi_normalize cost time: ",t2-t1)
    envi_normalize_list.append(t2-t1)
    with torch.no_grad():
        # imgData = imgData.permute(0, 2, 3, 1)
        imgData = torch.from_numpy(imgData).float().to(device)
        imgData = imgData.permute(2,0,1)
        # imgData = Variable(imgData).float().to(device)
        imgData = imgData.unsqueeze(0)#B C H W
        # imgData = imgData.unsqueeze(0)
        # inputData = imgData.permute(0,3,1,2)
        #切块变成批量测试
        t4 = time.time()
        # pix_data = torch.empty([0, 1, 11, 11, 9], dtype=torch.float).to(device)
        # for x in range(5, imgData.size()[2] - 5):
        #     # for y in range(5, imgData.size()[3] - 5):
        #     for y in range(5, 6):
        #         pix_data = torch.cat((pix_data, imgData[:, :, x - 5:x + 6, y - 5:y + 6,:]), 0)
        predict = model(imgData)
        # predict = torch.squeeze(predict)
        predict_ind = torch.argmax(predict, dim=1) # B：只有一个维度
        # predict_ind = predict_ind.reshape(imgData.size()[2]-10,1)
        t5 = time.time()
        print("predict  cost time: ",t5-t4)
        predict_list.append(t5-t4)

envi_normalize_list = np.array(envi_normalize_list)
print("envi_normalize cost mean time: ",np.mean(envi_normalize_list))

predict_list = np.array(predict_list)
print("predict  cost mean time: ",np.mean(predict_list))

        # print("predict 1849 * 1405 pixels cost time :",1405*(t5-t4))
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







