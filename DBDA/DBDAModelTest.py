import sys
sys.path.append('../')
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
# waterImgRootList = os.listdir(waterImgRootPath)
# waterImgRootList = [x for x in waterImgRootList if x[-4:] == '.img']
# waterImgRootPathList = ['vedio1', 'vedio2', 'vedio3', 'vedio4', 'vedio5', 'vedio6', 'vedio7']
waterImgRootPathList = ['train']#test
# select_bands = [2,36,54,61,77,82,87,91,95,104,108]
# select_bands = [x + 5 for x in  select_bands]
# select_bands = [116, 125, 109, 100, 108,  53,  98,  90,  81, 127, 123,  19]

select_bands = [2,36,54,61,77,82,87,91,95,104,108]
select_bands = [x + 5 for x in  select_bands]
# select_bands = [x for x in range(128)]
# imgpath
label_data_dir = '/home/cjl/dataset/label/'
png_path = '/home/cjl/ssd/dataset/shenzhen/rgb/needmark1/'
# png_path = 'E:/tmp/water/daytime/rgb/'

# csv2_save_path = log+'class4_allFile500_acc.csv'
# model_dict = {1:'ori_model_hz',2:'ori_model_hz'}
class_nums = 2
# model_path = "./IntervalSampleAddFeatureWaterModel_shenzhen/"
# model_path = "./small_32_0.001_True_True_False_sig/"
model_select = args.model_select
model_path = "./small_32_0.001_True_True_False_sig/"
LEN = 5
featureTrans = False
if featureTrans:
    inputBands = 21
else:
    inputBands = len(select_bands)

color_class = [[0,0,255],[255,0,0],[0,255,0]]
# epoch_list = [str(x) for x in [1,2,3,4,6,7,8,9,120,150,180,230,250,280,299]]
epoch_list = ['1']
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
    if model_select == 4:
        from DBDA.DBDA_net import DBDA_network_without_attention
        model = DBDA_network_without_attention(11, class_nums).cuda()
        model.load_state_dict(torch.load('/home/cjl/ssd/ywj/material/DBDA/4/89test_acc_0.9602631578947368.pkl'))
    elif model_select==5:
        from DBDA.DBDA_net import DBDA_network_three_losses
        model = DBDA_network_three_losses(11, 2).cuda()
        model.load_state_dict(torch.load('/home/cjl/ssd/ywj/material/DBDA/5/131test_acc_0.966.pkl'))
    elif model_select==6:
        from DBDA.DBDA_net import DBDA_network_three_losses_cross_Attention
        model = DBDA_network_three_losses_cross_Attention(11, 2).cuda()
        model.load_state_dict(torch.load('/home/cjl/ssd/ywj/material/DBDA/6/296test_acc_0.9515263157894737.pkl'))
    # 一定要加测试模式 有BN层或者dropout 的都需要，最好一直有
    # model.eval()
    model.eval()
    label_list = []
    predict_list = []
    count_right = 0
    count_tot = 0
    # result_dir = './resTrain/' + model_path[2:] + epoch + '/'
    result_dir = './resTrain/' + str(model_select) + '/'
    # result_dir_label = './res/'+ epoch + '/'
    # file_list = Shenzhen_test
    # file_list = waterFile
    file_list = SeaFile

    file_list = [x[3:] for x in file_list]
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
    for i in range(cnt):
        file_tmp = file_list[i*test_batch:(i+1)*test_batch if (i+1)<cnt else len(file_list)]
        imgData = []
        for filename in file_tmp:

            imgData_tmp = None
            if os.path.exists(waterImgRootPath + filename + '.img'):
                imgData_tmp = envi_loader(waterImgRootPath, filename, select_bands, False)
            else:
                for tmpImgPath in waterImgRootPathList:
                    if os.path.exists(waterImgRootPath + tmpImgPath + '/' + filename + '.img'):
                        imgData_tmp = envi_loader(waterImgRootPath + tmpImgPath + '/', filename, select_bands, False)
                        break
            # t3 = time.time()
            # 没必要 特征变换 增加之前设计的斜率特征

            if imgData_tmp is None:
                print("Not Found ", filename)
                continue
            if featureTrans:
                imgData_tmp = kindsOfFeatureTransformation(imgData_tmp)
            else:
                print("normalizing......")
                imgData_tmp = envi_wholeMaxnormalize(imgData_tmp)
            # imgData_tmp = envi_loader(env_data_dir, filename, True)
            # envi_loader(imgpath, file[3:], nora)
            # imgData_tmp = transform2(imgData_tmp)
            imgData.append(imgData_tmp)
        imgData = np.array(imgData)
        inputData = torch.tensor(imgData).float().cuda()
        #print(filename)
        # imgData = envi_loader(env_data_dir, filename)
        # imgData = transform2(imgData)
        # # imgData, imgData_cut = envi_loader_cut(env_data_dir, filename)
        # # imgData = transform_cut(imgData, imgData_cut)
        # imgData = imgData.transpose(2, 0, 1)
        # cnt+=1
        # if cnt<test_batch:
        #     imgData_tmp = np.load(npy_path + filename + '.npy')
        #     imgData.append(imgData_tmp)
        #     continue
        # else:

            # imgData_tmp = np.expand_dims(imgData_tmp, 0)
            # inputData_tmp = torch.tensor(imgData_tmp).float().cuda()

        with torch.no_grad():
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            try:
                inputData = inputData.permute(0,3,1,2)
            except Exception as info:
                print(info)
                continue
            print(inputData.shape)
            predict = model(inputData)
            del inputData
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            if model_select==4:
            # predict = torch.squeeze(predict)
                predict_ind = torch.argmax(predict, dim=1)
                predict_ind = predict_ind.cpu().detach().numpy()
                png_num = predict_ind.shape[0]
                for png_i in range(png_num):
                    png_path_single = png_path + "rgb" + file_tmp[png_i] + '.png'
                    print(png_path_single)
                    if not os.path.exists(png_path_single):
                        continue
                    imgGt = cv2.imread(png_path + "rgb" + file_tmp[png_i] + '.png')
                    imgGt = imgGt[5:-5,5:-5]
                    imgRes = imgGt
                    for color_num in [1,2]:
                        imgRes = mask_color_img(imgRes,mask=(predict_ind[png_i] == color_num),color=color_class[color_num-1],alpha=0.7 )
                    print(result_dir + file_tmp[png_i] + '.png')
                    cv2.imwrite(result_dir + file_tmp[png_i] + '.png', imgRes)
                    # cv2.imwrite(result_dir_label + file_tmp[png_i] + '.png', predict_ind[png_i])
                    gc.collect()
            else:
                for i in range(3):
                    predict_ind = torch.argmax(predict[i], dim=1)
                    predict_ind = predict_ind.cpu().detach().numpy()
                    png_num = predict_ind.shape[0]
                    for png_i in range(png_num):
                        png_path_single = png_path + "rgb" + file_tmp[png_i] + '.png'
                        print(png_path_single)
                        if not os.path.exists(png_path_single):
                            continue
                        imgGt = cv2.imread(png_path + "rgb" + file_tmp[png_i] + '.png')
                        imgGt = imgGt[5:-5, 5:-5]
                        imgRes = imgGt
                        for color_num in [1, 2]:
                            imgRes = mask_color_img(imgRes, mask=(predict_ind[png_i] == color_num),
                                                    color=color_class[color_num - 1], alpha=0.7)
                        print(result_dir + file_tmp[png_i] + '.png')
                        cv2.imwrite(result_dir + file_tmp[png_i] + '_' + str(i) + '.png', imgRes)
                        # cv2.imwrite(result_dir_label + file_tmp[png_i] + '.png', predict_ind[png_i])
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
