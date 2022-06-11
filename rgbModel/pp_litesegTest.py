import sys
sys.path.append('../')
from model_block.materialNet import MaterialSubModel
from model_block.PP_liteseg_final import PPLiteSeg
from model_block.materialNet import *
from utils.load_spectral import *
import gc
from utils.add_color import mask_color_img
from utils.accuracy_helper import *
from utils.os_helper import mkdir
import math
from utils.parse_args import parse_test_args
import os
import warnings
import torch.nn.functional as F
warnings.filterwarnings("ignore")
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'
# CUDA:0
args = parse_test_args()
FOR_TESTSET = args.FOR_TESTSET
test_batch = args.test_batch
print(FOR_TESTSET,test_batch)

file_str = {0:'_hz_train_',1:'_hz_test_',2:'_addsh_train_',3:'_addsh_test_'}
log = './log/'
mkdir(log)
# env_data_dir = 'D:/ZF2121133HHX/20220407/vedio3/'
env_data_dir = 'D:/ZF2121133HHX/20220407/vedio3/'
waterLabelPath = '/home/cjl/ssd/dataset/shenzhen/label/Label_rename/'
# waterImgRootPath = 'D:/ZF2121133HHX/water/daytime/'
waterImgRootPath = '/home/cjl/ssd/dataset/shenzhen/img/train/'
hangzhou_img_path = '/home/cjl/ssd/dataset/hangzhou/'
# waterImgRootList = os.listdir(waterImgRootPath)
# waterImgRootList = [x for x in waterImgRootList if x[-4:] == '.img']
# waterImgRootPathList = ['vedio1', 'vedio2', 'vedio3', 'vedio4', 'vedio5', 'vedio6', 'vedio7']
waterImgRootPathList = ['train']#test
# select_bands = [2,36,54,61,77,82,87,91,95,104,108]
# select_bands = [x + 5 for x in  select_bands]
select_train_bands = [123,  98, 114, 100, 109, 112, 108, 102, 81, 125, 53, 92]  # 模型选择结果波段
# select_bands = [116, 125, 109, 100, 108,  53,  98,  90,  81, 127, 123,  19]
# select_bands = [x for x in range(128)]
# imgpath
label_data_dir = '/home/cjl/dataset/label/'
png_path = '/home/cjl/ssd/dataset/shenzhen/rgb/needmark1/'
hz_png_path = '/home/cjl/ssd/dataset/hangzhou/rgb/'
# png_path = 'E:/tmp/water/daytime/rgb/'

# csv2_save_path = log+'class4_allFile500_acc.csv'
# model_dict = {1:'ori_model_hz',2:'ori_model_hz'}
class_nums = 2
# model_path = "./IntervalSampleAddFeatureWaterModel_shenzhen/"
# model_path = "./small_32_0.001_True_True_False_sig/"
model_path = './PPLiteSeg_Spectral_500000_0.001_4_1217/'
LEN = 5
featureTrans = False
if featureTrans:
    inputBands = 21
else:
    inputBands = len(select_train_bands)

color_class = [[0,0,255],[255,0,0],[0,255,0]]
epoch_list = [str(x) for x in [299, 198, 100,50,30,20,10]]

mean = torch.tensor([0.5, 0.5, 0.5]).cuda()
std = torch.tensor([0.5, 0.5, 0.5]).cuda()

# epoch_list = [str(x) for x in []]
# epoch_list = [str(x) for x in [299]]

# f2 = open(csv2_save_path, 'w', newline='')
# f2_csv = csv.writer(f2)
# csv2_header = ['epoch', 'micro accuracy','macro avg f1','other_pre','other_rec','other_f1',
#                'skin_pre', 'skin_rec','skin_f1','cloth_pre', 'cloth_rec','cloth_f1',
#                'plant_pre','plant_rec','plant_f1']
# csv2_header = ['micro accuracy']
# f2_csv.writerow(csv2_header)
print(inputBands)

for epoch in epoch_list:
# MaterialModel input size in training = (m, channels, length, length)  (length == 11)
#     model = MaterialSubModel(20, 4).cuda()
#     model = MaterialSubModel(in_channels=20, out_channels=4, kernel_size = 5,padding_mode='reflect',mid_channels_1=24, mid_channels_2=12, mid_channels_3=6).cuda()
    # model = MaterialSubModel(in_channels=20, out_channels=4, kernel_size = 3,padding_mode='reflect',mid_channels_1=32, mid_channels_2=16, mid_channels_3=8).cuda()
    # model.load_state_dict(torch.load(r"/home/yuwenjun/lab/multi-category-all/model-lr3-3ker-lrp/34.pkl"))
    # criterion=nn.MSELoss()
    # model = MaterialSubModel(in_channels=20, out_channels=4, kernel_size = 7,padding_mode='reflect',mid_channels_1=40, mid_channels_2=60, mid_channels_3=16).cuda()
    # model = MaterialSubModel(in_channels=inputBands, out_channels=class_nums).cuda()
    # model.load_state_dict(torch.load('./model/lr-4/' + epoch + '.pkl'))
    # model = MaterialBigModel(inputBands, class_nums,len_features = 32, mid_channel1 = 16).cuda()
    # model = MaterialSubModel(inputBands, class_nums).cuda()
    model = PPLiteSeg(num_classes=3, input_channel=inputBands).cuda()
    model.load_state_dict(torch.load(model_path + epoch + '.pkl'))
    model.eval()  # 测试
    # 一定要加测试模式 有BN层或者dropout 的都需要，最好一直有
    # model.eval()

    label_list = []
    predict_list = []
    count_right = 0
    count_tot = 0
    result_dir = './resTrain/' + model_path[2:] + epoch + '/'
    # result_dir_label = './res/'+ epoch + '/'
    # file_list = Shenzhen_test
    # file_list = waterFile
    file_list = waterFile + SeaFile

    # file_list = [x[3:] for x in file_list]
    print("the number of test file:",len(file_list))
    # if FOR_TESTSET == 0:
    #     file_list = trainFile_hz
    #     # result_dir = './output_mulpng_add_sh_data_ori_3ker_patch/' + epoch + '/test/'
    # elif FOR_TESTSET == 1:
    #     file_list = testFile_hz
    # elif FOR_TESTSET == 2:
    #     file_list = trainFile
    # elif FOR_TESTSET == 3:
    #     file_list = testFile
    # else:
    #     print("please check FOR_TESTSET !")
    #     break

    mkdir(result_dir)
    # mkdir(result_dir_label)
    cnt = math.ceil(len(file_list)/test_batch)
    print(test_batch, cnt)
    model.eval()  # 测试
    for i in range(cnt):
        file_tmp = file_list[i*test_batch:(i+1)*test_batch if (i+1)<cnt else len(file_list)]
        imgData = []
        for filename in file_tmp:
            # if os.path.exists(png_path + filename + '.png'):
            #     imgData_tmp = cv2.imread(png_path + filename + '.png')
            # else:
            #     imgData_tmp = cv2.imread(hz_png_path + filename + '.png')
            # imgData_tmp = imgData_tmp.astype(np.float64)[:, :, ::-1]
            # imgData.append(imgData_tmp)
            imgData_tmp = None
            if os.path.exists(waterImgRootPath + filename[3:] + '.img'):
                imgData_tmp = envi_loader(waterImgRootPath, filename[3:], select_train_bands, False)
            elif os.path.exists(hangzhou_img_path + filename[3:] + '.img'):
                imgData_tmp = envi_loader(hangzhou_img_path, filename[3:], select_train_bands, False)
            # # t3 = time.time()
            # # 没必要 特征变换 增加之前设计的斜率特征
            #
            if imgData_tmp is None:
                print("Not Found ", filename)
                continue
            if featureTrans:
                imgData_tmp = kindsOfFeatureTransformation(imgData_tmp)
            else:
                # if nora
                print("normalizing......")
                imgData_tmp = envi_wholeMaxnormalize(imgData_tmp)
                # imgData_tmp = envi_normalize(imgData_tmp)
            imgData.append(imgData_tmp)
        imgData = np.array(imgData)
        inputData = torch.tensor(imgData).float().cuda()
        # inputData = inputData / 255.0
        # inputData -= mean
        # inputData /= std
        inputData = inputData.permute(0, 3, 1, 2)
        with torch.no_grad():
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            print(inputData.shape)
            predict = model(inputData)
            del inputData
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            # predict = torch.squeeze(predict)
            # print(len(predict))
            # print(predict.shape)
            outadd = F.softmax(predict[0], dim=1)
            outadd = torch.argmax(outadd, dim=1)
            predadd = outadd.detach().cpu().numpy()
            predict_ind = np.int32(predadd)

            del predict
            del outadd
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            # predict_ind = torch.argmax(predict, dim=1)
            # predict_ind = predict_ind.cpu().detach().numpy()
        #print(np.unique(predict_ind))

        # generate result
        #     rows, cols = predict_ind.shape
            png_num = predict_ind.shape[0]
            for png_i in range(png_num):
                png_path_single = png_path + file_tmp[png_i] + '.png'
                # print(png_path_single)
                # if not os.path.exists(png_path_single):
                #     continue
                if os.path.exists(png_path + file_tmp[png_i]  + '.png'):
                    imgGt = cv2.imread(png_path + file_tmp[png_i]  + '.png')
                else:
                    imgGt = cv2.imread(hz_png_path + file_tmp[png_i]  + '.png')
                # imgGt = cv2.imread(png_path + file_tmp[png_i] + '.png')
                # imgGt = imgGt[5:-5,5:-5]

                imgRes = imgGt
                for color_num in [1]:
                    imgRes = mask_color_img(imgRes,mask=(predict_ind[png_i] == color_num),color=color_class[color_num-1],alpha=0.6 )
                # imgRes = np.zeros((rows, cols, 3), np.uint8)
                # imgRes[predict_ind == 1, 2] = 255
                # imgRes[predict_ind == 2, 0] = 255
                # imgRes[predict_ind == 3, 1] = 255
                # print(result_dir + file_tmp[png_i] + '.png')
                cv2.imwrite(result_dir + file_tmp[png_i] + '.png', imgRes)
                # cv2.imwrite(result_dir_label + file_tmp[png_i] + '.png', predict_ind[png_i])

                # img_label = cv2.imread(label_data_dir + file_tmp[png_i] + '.png', cv2.IMREAD_GRAYSCALE)
                # # img_label = img_label[5:5+rows, 5:5+cols]
                # img_label = img_label[5:-5,5:-5]
                # img_label = transformLabel(img_label, MUTI_CLASS_SH)

                # imgGt = cv2.imread(png_path+filename+'.png')
                # imgGt = np.zeros((rows, cols, 3), np.uint8)
                # imgGt = imgGt1
                # for color_num in [1,2]:
                #     imgGt = mask_color_img(imgGt,mask=(img_label == color_num),color=color_class[color_num-1],alpha=0.7 )
                # imgGt = mask_color_img(imgGt, mask=np.logical_or(img_label == 0 , img_label == 3), color=[125,125,125],
                #                            alpha=0.7)

                    # label_png = mask_color_img(label_png, label, color=class_color[i], alpha=0.7)
                # imgGt[img_label == 1, 2] = 255
                # imgGt[img_label == 2, 0] = 255
                # imgGt[img_label == 3, 1] = 255

                # cv2.imwrite(result_dir + file_tmp[png_i] + '_gt.png', imgGt)

                # calculate TN/TF/F-score and so on
                # predict_0, gt_0 = countPixels(img_label, 0, predict_ind[png_i], 0) #预测正确数量、真实标签数量
                # predict_1, gt_1 = countPixels(img_label, 1, predict_ind[png_i], 1)
                # predict_2, gt_2 = countPixels(img_label, 2, predict_ind[png_i], 2)
                # predict_3, gt_3 = countPixels(img_label, 3, predict_ind[png_i], 3)

                # label_list.append(img_label)
                # predict_list.append(predict_ind[png_i])
                # count_right += (predict_0 + predict_1 + predict_2 + predict_3)
                # count_tot += (gt_0 + gt_1 + gt_2 + gt_3)

                gc.collect()
    # label_list = np.array(label_list).flatten()
    # predict_list = np.array(predict_list).flatten()
    # ind = (label_list != 255)
    # predict_list = predict_list[ind]
    # label_list = label_list[ind]
    # # micro = f1_score(label_list, predict_list, average="micro")
    # # macro = f1_score(label_list, predict_list, average="macro")
    #
    # target_names = ['other','skin_','cloth','plant']
    # res = classification_report(label_list, predict_list, target_names=target_names,output_dict=True)
    # accuracy = count_right / count_tot
    # print('epoch ' + epoch + ' accuracy: ',accuracy)

    # csv2_line = [epoch, res['accuracy'], res['macro avg']['f1-score']]
    # # print('accuracy=', accuracy, 'micro=', micro, 'macro', macro)
    # for k in target_names:
    #     # print(k,'skin pre=', res['skin']['precision'], 'skin rec=', res['skin']['recall'])
    #     print(k, ' pre=', res[k]['precision'], ' rec=', res[k]['recall'], ' f1-score=', res[k]['f1-score'])
    # #     csv2_line.extend([res[k]['precision'], res[k]['recall'], res[k]['f1-score']])
    # print('all test micro accuracy:', res['accuracy'])
    # print('\n')
    # f2_csv.writerow([res['accuracy']])

    # MIoU
    # iou = np.sum(np.logical_and(label_list==predict_list,label_list!=0))
    # all_cnts = np.sum(label_list==1)+np.sum(label_list==2)+np.sum(label_list==3)
    # two_acc = iou/all_cnts
    # print("three acc:",two_acc)
    # gc.collect()
# f2.close()
# csv2_header = ['epoch', 'micro accuracy','macro avg f1','other_pre','other_rec','other_f1',
#                'skin_pre', 'skin_rec','skin_f1','cloth_pre', 'cloth_rec','cloth_f1',
#                'plant_pre','plant_rec','plant_f1']
