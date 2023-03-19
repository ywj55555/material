'''
每张图生成2250个测试像素，间隔取样2000个点左右（标注的点）展开成一维取样？？二维间隔取样才能保证稀疏性，更加模拟整图预测
网格取样需要每一类单独计算间隔，然后按照每一类别数量在整图的比例进行取样
'''
from tqdm import tqdm
import csv
import torch
from model_block import network
# from utils.load_spectral import envi_loader
from utils.load_spectral import *
from utils.accuracy_helper import *
from utils.os_helper import mkdir
# from sklearn.metrics import classification_report
from utils.load_spectral import kindsOfFeatureTransformation,envi_normalize
from sklearn.metrics import classification_report, cohen_kappa_score
from utils.parse_args import parse_test_args
import os
import math
import time
import skimage.io as io
from model_block.DBDA_Conv import DBDA_network_MISH_full_conv
from utils.add_color import mask_color_img
# from data.utilNetNew import modifySkinClothLabel
import sys

args = parse_test_args()
FOR_TESTSET = args.FOR_TESTSET
# FOR_TESTSET = 1
test_batch = args.test_batch
# test_batch = 4
model_select = args.model_select
# model_select = 1
print(FOR_TESTSET, test_batch)
print('model_select', model_select)
log = './log/'
mkdir(log)
all_label_path = '/home/cjl/dataset_18ch/label/'
all_png_path = '/home/cjl/dataset_18ch/raw_data/'
skinClothRawPath = '/home/cjl/dataset_18ch/raw_data/'
waterRawPath = '/home/cjl/dataset_18ch/waterBmh/'
testRawPath = '/home/cjl/dataset_18ch/needTestRaw/'
det_num = 20000
print('det_num ', det_num)
hand_selected_bands = [0, 1, 3, 7, 10, 13, 15, 16, 17]
labelToclass18skin = [255, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0]
class_nums = 4
cut_num = 10
model_root_path = '/home/cjl/ywj_code/graduationCode/alien_material/model/'
model_path = ['SkinClothWaterDBDA_0.001_64_4_handSelect_22276800',
              'SkinClothWater18_SSRN_0.001_8192_4_handSelect_22276800',
              'SkinClothWater18_DBMA_0.001_64_4_handSelect_22276800',
              'SkinClothWater18_FDSSC_0.001_8192_4_handSelect_22276800',
              ]
model_name = ['DBDA', 'SSRN', 'DBMA', 'FDSSC']
inputBands = len(hand_selected_bands)
test_file_type = ['train', 'test', 'extraTest', 'allTest', 'extraTest']
color_class = [[0, 0, 255], [255, 0, 0], [0, 255, 0]]

epoch_list = [str(x) for x in range(4, 300, 5)]  # twoBranchWhole

csv2_save_path = './log/'
mkdir(csv2_save_path)
f2 = open(csv2_save_path + model_name[model_select - 1] + '_skinClothWater18.csv', 'w', newline='')
f2_csv = csv.writer(f2)
csv2_header = ['epoch', 'micro accuracy', 'macro avg f1', 'kappa_score', 'other_pre', 'other_rec', 'other_f1',
               'skin_pre', 'skin_rec', 'skin_f1', 'cloth_pre', 'cloth_rec', 'cloth_f1',
               'water_pre', 'water_rec', 'water_f1']
# csv2_header = ['micro accuracy']
f2_csv.writerow(csv2_header)
print('inputBands', inputBands)


# if FOR_TESTSET == 1:
#     file_list = allTest18
# elif FOR_TESTSET == 2:
#     file_list = extraTest18
# else:
#     file_list = allTrain18

if FOR_TESTSET == 1:
    file_list = allTest18
elif FOR_TESTSET == 2:
    file_list = allExtraTest18
elif FOR_TESTSET == 3:
    file_list = extraTest18 + allTest18
elif FOR_TESTSET == 4:
    file_list = allExtraSelectedTest18
else:
    file_list = allTrain18
# file_list = [x[3:] for x in file_list]
print("the number of test file:",len(file_list))
cost_time = 0
for epoch in epoch_list:
    tmp_model_path = model_root_path + model_path[model_select - 1] + '/' + epoch + '.pkl'
    if not os.path.exists(tmp_model_path):
        continue
    label_list = []
    predict_list = []
    count_right = 0
    count_tot = 0
    if model_select == 1:
        model = network.DBDA_network_MISH(inputBands, class_nums).cuda()
    elif model_select == 2:  # 减少参数量！！！
        model = network.SSRN_network(inputBands, class_nums).cuda()
    elif model_select == 3:
        model = network.DBMA_network(inputBands, class_nums).cuda()
    elif model_select == 4:  # 减少参数量！！！
        model = network.FDSSC_network(inputBands, class_nums).cuda()
    else:
        print('model none !!!')
        sys.exit()
    result_dir = './result/' + model_name[model_select - 1] + '_skinClothWater18/' + test_file_type[FOR_TESTSET] + '/' + \
                 epoch + '/'
    result_label_dir = './result/' + model_name[model_select - 1] + '_skinClothWater18/' + test_file_type[FOR_TESTSET] + \
                       '_label/' + epoch + '/'
    mkdir(result_dir)
    mkdir(result_label_dir)
    model.load_state_dict(torch.load(tmp_model_path))
    # 一定要加测试模式 有BN层或者dropout 的都需要，最好一直有
    model.eval()
    t1 = time.time()
    with torch.no_grad():
        for filename in file_list:
            label_data = io.imread(all_label_path + filename + '.png')
            # 要先计算训练集的均值和方差！！
            imgData = None
            if os.path.exists(skinClothRawPath + filename + '.raw'):
                imgData = raw_loader(skinClothRawPath, filename, False, cut_num=cut_num,
                                         bands=hand_selected_bands)
            elif os.path.exists(waterRawPath + filename + '.raw'):
                imgData = raw_loader(waterRawPath, filename, False, cut_num=cut_num, bands=hand_selected_bands)
            elif os.path.exists(testRawPath + filename + '.raw'):
                imgData = raw_loader(testRawPath, filename, False, cut_num=cut_num, bands=hand_selected_bands)
            else:  # 990
                print(filename, ' raw not exist!!!')
            # t3 = time.time()
            # 没必要 特征变换 增加之前设计的斜率特征
            if imgData is None:
                print("Not Found ", filename)
                continue
            # H W 22
            tmp_cut = cut_num + 5  # 15
            if filename in allExtraTest18:
                tmp_cut -= 5  #10
            label_tmp = label_data[tmp_cut:-tmp_cut, tmp_cut:-tmp_cut]  # 990
            if filename in allExtraTest18:
                changeWaterLable(label_tmp)
            if filename in allWater18:
                label_tmp[label_tmp == 1] = 3
            if filename in allSkinCloth18:
                # modifySkinClothLabel(label_tmp)
                for i in range(3, 12):
                    label_tmp[label_tmp == i] = 10
                label_tmp[label_tmp == 0] = 255
                label_tmp[label_tmp == 10] = 0
            label_fla = label_tmp.flatten()
            tmp_w = label_tmp.shape[0]
            tmp_h = label_tmp.shape[1]
            print('label :', tmp_w, 'img: ',imgData.shape[0])
            label_coor = [[x, y] for x in range(tmp_w) for y in range(tmp_h)]
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
            single_pred_total = 0
            single_pred_correct = 0
            imgData = torch.from_numpy(imgData).float().cuda()
            labelData = torch.from_numpy(label_fla).long().cuda()

            imgData = imgData / torch.max(imgData, dim=2, keepdim=True)[0]
            # pred_total+=labelData.size()[0]
            # imgdata = imgdata.float().cuda()
            imgData = imgData.unsqueeze(0)  # B C H W D
            imgData = imgData.unsqueeze(0)

            cnt = math.ceil(len(label_coor)/det_num)
            predict_png = []
            for i in range(cnt):
                coor_tmp = label_coor[i * det_num:(i + 1) * det_num if (i + 1) < cnt else len(label_coor)]
                pix_data = torch.empty([0, 1, 11, 11, inputBands], dtype=torch.float).cuda()
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
                count_right += (predict_ind == labelData[i * det_num:(i + 1) * det_num if (i + 1) < cnt else len(label_coor)]).sum().item()

                predict_ind = predict_ind.cpu().numpy()
                predict_list.extend(predict_ind)
                predict_png.extend(predict_ind)
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()

            # if (int(epoch) + 1) % 10 != 0:
            #     continue
            predict_png = np.array(predict_png)
            predict_png = predict_png.reshape(imgData.size()[2]-10,imgData.size()[3]-10)

            imgGt = cv2.imread(all_png_path + filename + '.png')
            if filename in allExtraTest18:
                tmp_cut += 5  #15
            imgGt = imgGt[tmp_cut:-tmp_cut, tmp_cut:-tmp_cut]  # 990
            # imgGt2 = imgGt.copy()
            for color_num in range(1, class_nums):
                imgGt = mask_color_img(imgGt,mask=(predict_png == color_num),color=color_class[color_num-1],alpha=0.7 )

            cv2.imwrite(result_dir + filename+ '.png', imgGt)
            # 保存预测结果图像
            cv2.imwrite(result_label_dir + filename+ '.png', predict_png)

            del imgData
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
    t2 = time.time()
    single_cost = t2 - t1
    cost_time += single_cost
    # label_list[label_list=]
    label_list = np.array(label_list).flatten()
    predict_list = np.array(predict_list).flatten()
    remain_pix = label_list != 255
    label_list = label_list[remain_pix]
    predict_list = predict_list[remain_pix]

    target_names = ['other', 'skin_', 'cloth', 'water']

    res = classification_report(label_list, predict_list, target_names=target_names, output_dict=True)
    kappa_score = cohen_kappa_score(label_list, predict_list)  # 计算卡帕系数
    # accuracy = count_right / count_tot

    csv2_line = [epoch, res['accuracy'], res['macro avg']['f1-score'], kappa_score]
    # print('accuracy=', accuracy, 'micro=', micro, 'macro', macro)
    for k in target_names:
        # print(k,'skin pre=', res['skin']['precision'], 'skin rec=', res['skin']['recall'])
        print(k, ' pre=', res[k]['precision'], ' rec=', res[k]['recall'], ' f1-score=', res[k]['f1-score'])
        csv2_line.extend([res[k]['precision'], res[k]['recall'], res[k]['f1-score']])
    print(epoch, 'all test micro accuracy:', res['accuracy'], 'macro avg :', res['macro avg']['f1-score'],
          'kappa_score: ', kappa_score)
    print('\n')
    print('average cost time :', single_cost / len(file_list))
    f2_csv.writerow(csv2_line)
f2.close()
print('all cost time:', cost_time)










