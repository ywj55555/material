'''
每张图生成2250个测试像素，间隔取样2000个点左右（标注的点）展开成一维取样？？二维间隔取样才能保证稀疏性，更加模拟整图预测
网格取样需要每一类单独计算间隔，然后按照每一类别数量在整图的比例进行取样
'''
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

waterLabelPath = 'D:/dataset/lg/Label_rename/'
waterImgRootPath = 'D:/dataset/lgimg/train/'
png_path = 'D:/dataset/lg/needmark1/'
log = './log/'
mkdir(log)
det_num = 10000
train_mean = np.array([1638.30293365, 1856.07645628, 2395.36102414, 2963.61267338, 3825.72213097,
 3501.42091296, 4251.15954562, 3965.7991342,  3312.6616123 ])
# train_mean = np.array(train_mean)
train_var = np.array([ 5402022.45076405,  6679240.29294402,  9629853.43114098, 15170998.1931259,
 25767233.41870246, 18716092.38116236, 18231725.96313715, 28552394.38170321,
 13127436.03417886])
train_mean = torch.from_numpy(train_mean).float().cuda()
train_var = torch.from_numpy(train_var).float().cuda()
# train_var = np.array(train_var)
color_class = [[0,0,255],[255,0,0],[0,255,0]]

args = parse_test_args()
model_select = args.model_select
print('model select: ',model_select)
model_str = {0:'_DBDA_',1:'_SSRN_'}
epoch = str(10)
model_list = ['D:/ZY2006224YWJ/python/ori_multi-category/DBDA_model_32_0.001/10.pkl',
              '/home/cjl/ywj_code/contrast_model_sh/SSRN_model_cut_scale32_0.0005/24.pkl']
bands = 11
CLASSES_NUM = 2
nora = True
select_bands = [2,36,54,61,77,82,87,91,95,104,108]
device = torch.device('cuda')
if model_select == 1:
    model = network.DBDA_network_MISH(bands, CLASSES_NUM).to(device)
else:
    model = DBDA_network_MISH_full_conv(bands, CLASSES_NUM, 4).to(device)
    # model = network.SSRN_network(bands, CLASSES_NUM).to(device)
model_path = model_list[model_select - 1]
model.load_state_dict(torch.load(model_path))
model.eval()

csv2_save_path = log+'pix_predict'+model_str[model_select]+'dfx_testfile_data.csv'
f2 = open(csv2_save_path, 'w', newline='')
f2_csv = csv.writer(f2)
# csv2_header = ['micro accuracy','three_acc']
csv2_header = ['micro accuracy']
f2_csv.writerow(csv2_header)
featureTrans = False

result_dir = './res/'+model_str[model_select]+'pre_label'+'/'
result_pred = './resTrain/' + "DBDA/" + epoch + '/'
mkdir(result_dir)
mkdir(result_pred)
output_log = open('./log/output_test_DBDA.log','w')

label_list = []
predict_list = []
t1 = time.time()
# tmpfile = [x for x in alltrainFile if x not in dfx_test]
file_list = SeaFile
# file_list = waterFile

# file_list = [x[3:] for x in file_list]
print("the number of test file:",len(file_list))
with torch.no_grad():
    for file in file_list:
        label_data = cv2.imread(waterLabelPath + file+'.png',cv2.IMREAD_GRAYSCALE)
        # 要先计算训练集的均值和方差！！
        imgData = None
        if os.path.exists(waterImgRootPath + file[3:] + '.img'):
            imgData = envi_loader(waterImgRootPath, file[3:], select_bands, False)
        else:
            continue
        # t3 = time.time()
        # 没必要 特征变换 增加之前设计的斜率特征
        if imgData is None:
            print("Not Found ", file)
            continue
        # H W 22
        if featureTrans:
            print("kindsOfFeatureTransformation......")
            output_log.writelines("kindsOfFeatureTransformation......")
            # 11 -》 21
            imgData = kindsOfFeatureTransformation(imgData, nora)
        else:
            if nora:
                print("normalizing......")
                output_log.writelines("normalizing......")
                imgData = envi_normalize(imgData)

        label_tmp = label_data[5:-5,5:-5]
        label_tmp = transformLabel(label_tmp, MUTI_CLASS_WATER) #255,0,1,2,3
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
        imgData = torch.from_numpy(imgData).float().cuda()
        labelData = torch.from_numpy(label_fla).long().cuda()


        # pred_total+=labelData.size()[0]
        # imgdata = imgdata.float().cuda()
        imgData = imgData.unsqueeze(0)  # B C H W D
        imgData = imgData.unsqueeze(0)

        cnt = math.ceil(len(label_coor)/det_num)
        predict_png = []
        for i in tqdm(range(cnt)):
            coor_tmp = label_coor[i * det_num:(i + 1) * det_num if (i + 1) < cnt else len(label_coor)]
            # imgData = []
            pix_data = torch.empty([0, 1, 11, 11, bands], dtype=torch.float).cuda()
            for coor in coor_tmp:
                pix_data = torch.cat((pix_data, imgData[:, :, coor[0] :coor[0]+ 11, coor[1]:coor[1] +11, :]), 0)
        # pix_data_ = pix_data.permute()
        #     切面归一化：
        #     pix_data_ = pix_data.view(np.prod(pix_data.shape[:4]),pix_data.shape[4])
        #
        #     pix_data_ = (pix_data_-train_mean)/torch.sqrt(train_var)
        #
        #     pix_data = pix_data_.view(pix_data.shape)
            predict = model(pix_data)

        # print(predict.shape[0])
        # predict = torch.squeeze(predict)
            predict_ind = torch.argmax(predict, dim=1)  # B：只有一个维度
            del predict
            del pix_data
            predict_ind = predict_ind.squeeze() #B
            pred_correct += (predict_ind == labelData[i * det_num:(i + 1) * det_num if (i + 1) < cnt else len(label_coor)]).sum().item()

            predict_ind = predict_ind.cpu().numpy()
            predict_list.extend(predict_ind)
            predict_png.extend(predict_ind)
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()

        predict_png = np.array(predict_png)
        predict_png = predict_png.reshape(imgData.size()[2]-10,imgData.size()[3]-10)

        imgGt = cv2.imread(png_path + file + '.png')
        imgGt = imgGt[5:-5,5:-5]
        # imgGt2 = imgGt.copy()
        for color_num in [1]:
            imgGt = mask_color_img(imgGt,mask=(predict_png == color_num),color=color_class[color_num-1],alpha=0.7 )
        cv2.imwrite(result_pred + file + '.png', imgGt)

        del imgData
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()

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

# target_names = ['other','skin_','cloth','plant']
# res = classification_report(label_list, predict_list, target_names=target_names,output_dict=True)
#
csv2_line = [all_acc]
# # print('accuracy=', accuracy, 'micro=', micro, 'macro', macro)
# for k in target_names:
#     # print(k,'skin pre=', res['skin']['precision'], 'skin rec=', res['skin']['recall'])
#     print(k, ' pre=', res[k]['precision'], ' rec=', res[k]['recall'], ' f1-score=', res[k]['f1-score'])
#     csv2_line.extend([res[k]['precision'], res[k]['recall'], res[k]['f1-score']])
# print('all test micro accuracy:', res['accuracy'], ' all test macro avg f1:', res['macro avg']['f1-score'])
f2_csv.writerow(csv2_line)
f2.close()
output_log.close()









