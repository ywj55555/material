from model_block.materialNet import *
from utils.load_spectral import *
import gc
from utils.add_color import mask_color_img
from utils.accuracy_helper import *
from utils.os_helper import mkdir
import math
from sklearn.metrics import classification_report, f1_score
from utils.parse_args import parse_test_args
import os
import warnings

warnings.filterwarnings("ignore")
import csv

# os.environ['CUDA_VISIBLE_DEVICES'] = '6'
# CUDA:0
args = parse_test_args()
FOR_TESTSET = args.FOR_TESTSET
test_batch = args.test_batch
print(FOR_TESTSET, test_batch)
test_batch = 5
file_str = {0: '_hz_train_', 1: '_hz_test_', 2: '_addsh_train_', 3: '_addsh_test_'}
log = './log/'
mkdir(log)
cut_num = 10
labelpath = '/home/cjl/spectraldata/trainLabelAddWater/'
imgpath = '/home/cjl/spectraldata/RiverLakeTrainData/'

select_bands = [1, 13, 25, 52, 76, 92, 99, 105, 109]
code2labelSh = [255,2,2,2,0,2,2,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# select_bands = [x for x in range(128)]
# imgpath
# label_data_dir = '/home/cjl/dataset/label/'
# png_path = '/home/cjl/ssd/dataset/HIK/shuichi/rgb/'
png_path = '/home/cjl/spectraldata/water_skin_rgb/'
# png_path = 'E:/tmp/water/daytime/rgb/'

# csv2_save_path = log+'class4_allFile500_acc.csv'
# model_dict = {1:'ori_model_hz',2:'ori_model_hz'}
class_nums = 4
# model_path = "./IntervalSampleAddFeatureWaterModel_shenzhen/"
# model_path = "./small_32_0.001_True_True_False_sig/"
model_path = "/home/cjl/ywj_code/graduationCode/alien_material/model/HZWaterSkinAddSh_64_0.001_True_False_False/"
LEN = 5
featureTrans = False
if featureTrans:
    inputBands = 21
else:
    inputBands = len(select_bands)
# inputBands = 2
color_class = [[0, 0, 255], [255, 0, 0], [0, 255, 0]]
epoch_list = [str(x) for x in [295, 298, 282, 270, 190, 227, 277, 168, 268,
                               163, 200, 145, 253, 234, 247, 176, 177, 158, 208, 299]]

csv2_save_path = './log/'
mkdir(csv2_save_path)
f2 = open(csv2_save_path + 'HZWaterSkinAddShAddSH_SZ_HZWaterTest.csv', 'w', newline='')
f2_csv = csv.writer(f2)
csv2_header = ['epoch', 'micro accuracy', 'macro avg f1', 'other_pre', 'other_rec', 'other_f1',
               'skin_pre', 'skin_rec', 'skin_f1', 'cloth_pre', 'cloth_rec', 'cloth_f1',
               'water_pre', 'water_rec', 'water_f1']
# csv2_header = ['micro accuracy']
f2_csv.writerow(csv2_header)
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

    model = MaterialSubModel(inputBands, class_nums).cuda()
    model.load_state_dict(torch.load(model_path + epoch + '.pkl'))
    # 一定要加测试模式 有BN层或者dropout 的都需要，最好一直有
    model.eval()

    label_list = []
    predict_list = []
    count_right = 0
    count_tot = 0
    result_dir = './result/HZWaterSkinAddSh/test/' + epoch + '/'
    result_label_dir = './result/HZWaterSkinAddSh/testLabel/' + epoch + '/'
    file_list = HZRiverSkinClothTest + SH_Test + SZ_Test

    print("the number of test file:", len(file_list))
    mkdir(result_dir)
    mkdir(result_label_dir)
    # mkdir(result_dir_label)
    cnt = math.ceil(len(file_list) / test_batch)
    print(test_batch, cnt)

    for i in range(cnt):
        file_tmp = file_list[i * test_batch:(i + 1) * test_batch if (i + 1) < cnt else len(file_list)]
        imgData = []
        labelData = []
        for filename in file_tmp:
            imgData_tmp = None
            if os.path.exists(imgpath + filename + '.hdr'):
                # imgData_tmp = raw_loader(bmh_raw_path, filename, cut_num=cut_num)
                imgLabel_tmp = io.imread(labelpath + filename + '.png')
                if filename in SH_Test:
                    imgLabel_tmp = transformLabel(imgLabel_tmp, code2labelSh)
                elif filename in SZ_Test:
                    imgLabel_tmp[imgLabel_tmp == 0] = 255
                    imgLabel_tmp[imgLabel_tmp == 1] = 3
                    imgLabel_tmp[imgLabel_tmp == 2] = 0
                # imgLabel_tmp = imgLabel_tmp[cut_num:-cut_num, cut_num:-cut_num]
                imgLabel_tmp = imgLabel_tmp[5:-5, 5:-5]

                imgData_tmp = envi_loader(imgpath, filename, select_bands, True)
            else:
                print("Not Found ", filename)
                continue
            imgData.append(imgData_tmp)
            labelData.append(imgLabel_tmp)
        imgData = np.array(imgData)
        label_list.extend(labelData)
        inputData = torch.tensor(imgData).float().cuda()
        labelData = np.array(labelData)
        labelData = labelData.astype(np.uint8)
        labelData = torch.tensor(labelData).long().cuda()

        with torch.no_grad():
            torch.cuda.empty_cache()
            # torch.cuda.empty_cache()
            # torch.cuda.empty_cache()
            # torch.cuda.empty_cache()
            # torch.cuda.empty_cache()
            try:
                inputData = inputData.permute(0, 3, 1, 2)  # b c h w
            except Exception as info:
                print(info)
                continue
            # print(inputData.shape)
            # inputData = inputData / torch.max(inputData, dim=1, keepdim=True)[0]
            start = time.time()
            # inputData = inputData[:, select_bands, :, :]
            predict = model(inputData)
            # del inputData
            # torch.cuda.empty_cache()
            # torch.cuda.empty_cache()

            # predict = torch.squeeze(predict)
            predict_ind = torch.argmax(predict, dim=1)
            count_right += torch.sum(predict_ind == labelData).item()  # 全部训练
            # count_right += torch.sum((predictIndex == label) & (label != 0)).item()
            count_tot += torch.sum(labelData != 255).item()  # 一定是255 和前面需要对应起来
            predict_ind = predict_ind.cpu().detach().numpy()

            end = time.time()
            print("average cost time : ", (end - start) / predict_ind.shape[0])

            predict_list.extend(predict_ind)
            # print(np.unique(predict_ind))
            # 渲染
            png_num = predict_ind.shape[0]
            for png_i in range(png_num):
                png_path_single = png_path + file_tmp[png_i] + '.png'
                if not os.path.exists(png_path_single):
                    continue
                imgGt = cv2.imread(png_path + file_tmp[png_i] + '.png')
                # imgGt = imgGt[cut_num:-cut_num, cut_num:-cut_num,:]
                imgGt = imgGt[5:-5, 5:-5]

                imgRes = imgGt
                for color_num in range(1, class_nums):
                    imgRes = mask_color_img(imgRes, mask=(predict_ind[png_i] == color_num),
                                            color=color_class[color_num - 1], alpha=0.7)
                cv2.imwrite(result_dir + file_tmp[png_i] + '.png', imgRes)
                # 保存预测结果图像
                cv2.imwrite(result_label_dir + file_tmp[png_i] + '.png', predict_ind[png_i])
        # gc.collect()
    accuracy = count_right / count_tot
    print('epoch ' + epoch + ' accuracy: ', accuracy)

    label_list = np.array(label_list).flatten()
    predict_list = np.array(predict_list).flatten()
    ind = (label_list != 255)  # 一定要加这个
    predict_list = predict_list[ind]
    label_list = label_list[ind]
    # micro = f1_score(label_list, predict_list, average="micro")
    # macro = f1_score(label_list, predict_list, average="macro")

    target_names = ['other', 'skin_', 'cloth', 'water']
    res = classification_report(label_list, predict_list, target_names=target_names, output_dict=True)
    # metrics.cohen_kappa_score(pred_test_fdssc, gt_test[:-VAL_SIZE])  # 计算卡帕系数
    # accuracy = count_right / count_tot
    # print('epoch ' + epoch + ' accuracy: ',accuracy)

    csv2_line = [epoch, res['accuracy'], res['macro avg']['f1-score']]
    # print('accuracy=', accuracy, 'micro=', micro, 'macro', macro)
    for k in target_names:
        # print(k,'skin pre=', res['skin']['precision'], 'skin rec=', res['skin']['recall'])
        print(k, ' pre=', res[k]['precision'], ' rec=', res[k]['recall'], ' f1-score=', res[k]['f1-score'])
        csv2_line.extend([res[k]['precision'], res[k]['recall'], res[k]['f1-score']])
    print(epoch, 'all test micro accuracy:', res['accuracy'], 'macro avg :', res['macro avg']['f1-score'])
    print('\n')
    f2_csv.writerow(csv2_line)
f2.close()
# csv2_header = ['epoch', 'micro accuracy','macro avg f1','other_pre','other_rec','other_f1',
#                'skin_pre', 'skin_rec','skin_f1','cloth_pre', 'cloth_rec','cloth_f1',
#                'plant_pre','plant_rec','plant_f1']
