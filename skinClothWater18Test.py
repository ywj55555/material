from model_block.materialNet import *
from utils.load_spectral import *
import gc
from utils.add_color import mask_color_img
from utils.accuracy_helper import *
from utils.os_helper import mkdir
import math
from sklearn.metrics import classification_report, cohen_kappa_score
from utils.parse_args import parse_test_args
import os
import warnings
from model_block.spaceSpectrumFusionNet import spaceSpectrumFusionNet, SpectrumNet
from model_block.PP_liteseg_final import PPLiteSeg
from model_block.FreeNet import FreeNet
from model_block.SSDGL import SSDGL
from model_block.BiSeNetV2 import BiSeNetV2
from utils.paint_rect import dilate_open
import sys

warnings.filterwarnings("ignore")
import csv

args = parse_test_args()
FOR_TESTSET = args.FOR_TESTSET
# FOR_TESTSET = 1
test_batch = args.test_batch
# test_batch = 4
model_select = args.model_select
dilate_open_used = args.dilate_open
# model_select = 1
print(FOR_TESTSET, test_batch)
print(model_select)
print('dilate_open_used ', dilate_open_used)
log = './log/'
mkdir(log)
cut_num = 10
all_label_path = '/home/cjl/dataset_18ch/label/'
all_png_path = '/home/cjl/dataset_18ch/raw_data/'
skinClothRawPath = '/home/cjl/dataset_18ch/raw_data/'
waterRawPath = '/home/cjl/dataset_18ch/waterBmh/'
testRawPath = '/home/cjl/dataset_18ch/needTestRaw/'

hand_selected_bands = [0, 1, 3, 7, 10, 13, 15, 16, 17]
labelToclass18skin = [255, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0]
class_nums = 4
model_root_path = '/home/cjl/ywj_code/graduationCode/alien_material/model/'
model_path = ['SkinClothWater18_twoBranch_0.001_64_4_handSelect_22276800',
              'sinkClothWater18_PPLiteSeg_500000_0.001_16',
              'sinkClothWater18_FreeNet_500000_0.0008_4',
              'sinkClothWater18_SSDGL_500000_0.001_2',
              'sinkClothWater18_twoBranchWhole_500000_0.001_8',
              'SkinClothWater18_smallMolde_0.001_64_4_handSelect_22276800',
              'sinkClothWater18_BiSeNetv2_500000_0.001_10',
              'SkinClothWater18_twoBranch2.0_11_0.001_64_4_handSelect_22276800',
              'SkinClothWater18_twoBranch3.0_11_0.001_64_4_handSelect_22276800',
              'SkinClothWater18_twoBranch3.0_9_0.001_64_4_handSelect_22276800',
              'SkinClothWater18_twoBranch3.0_7_0.001_64_4_handSelect_22276800',
              'SkinClothWater18_twoBranch3.0_5_0.001_64_4_handSelect_22276800',
              'SkinClothWater18_onlySpaceBranch3.0_11_0.001_64_4_handSelect_22276800',
              'SkinClothWater18_onlySpectrumBranch3.0_11_0.001_64_4_handSelect_22276800',
              ]
model_name = ['twoBranch', 'PPLiteSeg', 'FreeNet', 'SSDGL', 'twoBranchWhole',
              'smallModel', 'BiSeNetv2', 'twoBranch2.0', 'twoBranch3.0',
              'twoBranch3.0_9', 'twoBranch3.0_7', 'twoBranch3.0_5',
              'twoBranch3.0_onlySpace', 'twoBranch3.0_onlySpectrum', ]  # 注意是否是2.0版本
inputBands = len(hand_selected_bands)
test_file_type = ['train', 'test', 'extraTest', 'allTest', 'extraTest']
color_class = [[0, 0, 255], [255, 0, 0], [0, 255, 0]]
mean = torch.tensor([0.5, 0.5, 0.5]).cuda()
std = torch.tensor([0.5, 0.5, 0.5]).cuda()
# epoch_list = [str(x) for x in [33, 53, 103, 23, 13, 6, 190, 150, 130]]  # twoBranch  每隔5个epoch来一次
# epoch_list = [str(x) for x in [54, 299, 154]]  # PPLiteSeg
# epoch_list = [str(x) for x in [19, 39, 44, 9]]  # FreeNet
# epoch_list = [str(x) for x in [94, 74, 54, 24, 14]]  # SSDGL
# epoch_list = [str(x) for x in [11, 9, 5]]  # twoBranchWhole
# epoch_list = [str(x) for x in [299, 199, 99, 59, 9]]  # smallModel
# epoch_list = [str(x) for x in [189, 129, 99, 59, 29, 19]]  # BiSeNetv2
# int(epoch) + 1) % 10 != 0
if model_select == 9:
    epoch_list = [str(x) for x in range(300)]
else:
    epoch_list = [str(x) for x in range(4, 300, 5)]
# epoch_list = [str(x) for x in [17]]

csv2_save_path = './log/'
mkdir(csv2_save_path)
f2 = open(csv2_save_path + model_name[model_select - 1] + test_file_type[FOR_TESTSET] + '_skinClothWater18.csv', 'w',
          newline='')
f2_csv = csv.writer(f2)
csv2_header = ['epoch', 'micro accuracy', 'macro avg f1', 'kappa_score', 'other_pre', 'other_rec', 'other_f1',
               'skin_pre', 'skin_rec', 'skin_f1', 'cloth_pre', 'cloth_rec', 'cloth_f1',
               'water_pre', 'water_rec', 'water_f1']
# csv2_header = ['micro accuracy']
f2_csv.writerow(csv2_header)
print(inputBands)
patchsize = 11
for epoch in epoch_list:
    tmp_model_path = model_root_path + model_path[model_select - 1] + '/' + epoch + '.pkl'
    if not os.path.exists(tmp_model_path):
        continue
    if model_select in [1, 5, 8, 9, 10, 11, 12, 13]:  # 5是Whole # 注意是否是2.0版本
        version = 2
        spectrum_used = True
        if model_select in [9, 10, 11, 12, 13]:
            version = 3
            if model_select == 10:
                patchsize = 9
            elif model_select == 11:
                patchsize = 7
            elif model_select == 12:
                patchsize = 5
            elif model_select == 13:
                spectrum_used = False
        model = spaceSpectrumFusionNet(inputBands, class_nums, patch_size=patchsize, spectrum_used=spectrum_used,
                                       version=version).cuda()
    elif model_select == 14:  # SpectrumNet
        model = SpectrumNet(inputBands, class_nums, patch_size=11, version=3).cuda()
    elif model_select == 2:  # PPLiteSeg
        cut_num = 0
        model = PPLiteSeg(num_classes=class_nums, input_channel=3).cuda()
    elif model_select == 3:  # FreeNet
        cut_num = 6
        model = FreeNet(bands=inputBands, class_nums=class_nums).cuda()
    elif model_select == 4:  # SSDGL
        cut_num = 6
        model = SSDGL(bands=inputBands, class_nums=class_nums).cuda()
    elif model_select == 6:  # smallModel
        model = MaterialSubModel(inputBands, class_nums).cuda()
    elif model_select == 7:  # BisNetv2
        cut_num = 0
        model = BiSeNetV2(n_classes=class_nums, aux_mode='train').cuda()
    else:
        print('model none!!')
        model = None
        sys.exit()
        # model = MaterialSubModel(inputBands, class_nums).cuda()

    model.load_state_dict(torch.load(tmp_model_path))
    # 一定要加测试模式 有BN层或者dropout 的都需要，最好一直有
    model.eval()
    if model_select == 7:
        model.aux_mode = 'eval'
    label_list = []
    predict_list = []
    predict_dilate_open_list = []
    count_right = 0
    count_right_dilate_open = 0
    count_tot = 0
    # 渲染测试集！
    result_dir = './result/' + model_name[model_select - 1] + '_skinClothWater18/' + test_file_type[FOR_TESTSET] + '/' + \
                 epoch + '/'
    result_label_dir = './result/' + model_name[model_select - 1] + '_skinClothWater18/' + test_file_type[FOR_TESTSET] + \
                       '_label/' + epoch + '/'
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
    print("the number of test file:", len(file_list))
    mkdir(result_dir)
    mkdir(result_label_dir)
    cnt = math.ceil(len(file_list) / test_batch)
    print(test_batch, cnt)
    for i in range(cnt):
        file_tmp = file_list[i * test_batch:(i + 1) * test_batch if (i + 1) < cnt else len(file_list)]
        imgData = []
        labelData = []
        tmp_cut_num = 0
        for filename in file_tmp:
            imgData_tmp = None
            if model_select not in [2, 7]:
                if os.path.exists(skinClothRawPath + filename + '.raw'):
                    imgData_tmp = raw_loader(skinClothRawPath, filename, False, cut_num=cut_num,
                                             bands=hand_selected_bands)
                elif os.path.exists(waterRawPath + filename + '.raw'):
                    imgData_tmp = raw_loader(waterRawPath, filename, False, cut_num=cut_num, bands=hand_selected_bands)
                elif os.path.exists(testRawPath + filename + '.raw'):
                    imgData_tmp = raw_loader(testRawPath, filename, False, cut_num=cut_num, bands=hand_selected_bands)
                else:
                    print(filename, ' raw not exist!!!')
            else:
                imgData_tmp = cv2.imread(all_png_path + filename + '.png')  # 加载模式为 BGR
                imgData_tmp = imgData_tmp.astype(np.float64)[:, :, ::-1]  # 转为 RGB 进行训练
                if model_select == 7:
                    imgData_tmp2 = np.zeros([imgData_tmp.shape[0] + 4, imgData_tmp.shape[1] + 4, 3], dtype=np.float32)
                    # imgData_tmp = np.zeros_like(imgData, dtype=np.float32)
                    imgData_tmp2[2:-2, 2:-2, :] = imgData_tmp
                    imgData_tmp = imgData_tmp2

            # imgLabel = imgLabel[cut_num:-cut_num, cut_num:-cut_num]
            imgLabel_tmp = io.imread(all_label_path + filename + '.png')

            if model_select in [1, 5, 6, 8, 9, 10, 11, 12, 13, 14]:
                tmp_cut_num = cut_num + int(patchsize / 2)
                if filename in allExtraTest18:
                    tmp_cut_num -= 5
                imgLabel_tmp = imgLabel_tmp[tmp_cut_num:-tmp_cut_num, tmp_cut_num:-tmp_cut_num]
            elif model_select in [3, 4]:
                tmp_cut_num = cut_num
                if filename in allExtraTest18:
                    tmp_cut_num -= 5
                imgLabel_tmp = imgLabel_tmp[tmp_cut_num:-tmp_cut_num, tmp_cut_num:-tmp_cut_num]
            elif model_select in [2] and filename in allExtraTest18:
                imgLabel_tmp2 = np.zeros([imgLabel_tmp.shape[0] + 10, imgLabel_tmp.shape[1] + 10], dtype=np.uint8)
                imgLabel_tmp2[5:-5, 5:-5] = imgLabel_tmp
                imgLabel_tmp = imgLabel_tmp2
            elif model_select in [7]:
                expand_size = 2
                if filename in allExtraTest18:
                    expand_size += 5
                imgLabel_tmp2 = np.zeros(
                    [imgLabel_tmp.shape[0] + expand_size * 2, imgLabel_tmp.shape[1] + expand_size * 2], dtype=np.uint8)
                imgLabel_tmp2[expand_size:-expand_size, expand_size:-expand_size] = imgLabel_tmp
                imgLabel_tmp = imgLabel_tmp2

            if filename in allExtraTest18:
                changeWaterLable(imgLabel_tmp)
            if filename in allWater18:
                imgLabel_tmp[imgLabel_tmp == 1] = 3
            if filename in allSkinCloth18:
                modifySkinClothLabel(imgLabel_tmp)
                for i in range(3, 12):
                    imgLabel_tmp[imgLabel_tmp == i] = 10
                imgLabel_tmp[imgLabel_tmp == 0] = 255
                imgLabel_tmp[imgLabel_tmp == 10] = 0

            imgData.append(imgData_tmp)
            # print(imgData_tmp.shape)
            labelData.append(imgLabel_tmp)
        imgData = np.array(imgData)
        label_list.extend(labelData)
        inputData = torch.tensor(imgData).float().cuda()
        labelData = np.array(labelData)
        labelData = labelData.astype(np.uint8)
        labelData = torch.tensor(labelData).long().cuda()
        with torch.no_grad():
            torch.cuda.empty_cache()
            if model_select in [2, 7]:
                inputData = inputData / 255.0
                inputData -= mean
                inputData /= std
            try:
                inputData = inputData.permute(0, 3, 1, 2)  # b c h w
            except Exception as info:
                print(info)
                continue
            spaceData = None
            start = time.time()
            if model_select in [1, 5, 8, 9, 10, 11, 12, 13]:  # twoBranch
                spaceData = inputData.clone()
                img_spac_max = spaceData.clone()
                tup = (2, 3)
                # 切面方向归一化
                for tdim in tup:
                    img_spac_max = torch.max(img_spac_max, dim=tdim, keepdim=True)[0]
                spaceData = spaceData / img_spac_max
            if model_select not in [2, 7]:
                inputData = inputData / torch.max(inputData, dim=1, keepdim=True)[0]
                # inputData = inputData.permute(0, 3, 1, 2)
            if model_select in [1, 5, 8, 9, 10, 11, 12, 13]:
                predict = model(inputData, spaceData)
            # elif model_select == 14:
            #     predict = model(inputData)
            else:
                predict = model(inputData)

            if model_select != 2:
                # print(type(predict))
                # print(len(predict))
                predict_ind = torch.argmax(predict, dim=1)
            else:
                predict_ind = torch.argmax(predict[0], dim=1)  # 尽量让Bisenet也这样，部署的时候直接修改返回值即可！


            count_right += torch.sum(predict_ind == labelData).item()  # 全部训练
            # count_right += torch.sum((predictIndex == label) & (label != 0)).item()
            count_tot += torch.sum(labelData != 255).item()  # 一定是255 和前面需要对应起来
            predict_ind = predict_ind.cpu().detach().numpy()

            labelData = labelData.cpu().detach().numpy()
            end = time.time()
            # print("average cost time : ", (end - start) / predict_ind.shape[0])

            predict_list.extend(predict_ind)
            # print(np.unique(predict_ind))
            # if (int(epoch) + 1) % 10 != 0 and model_select != 9:
            #     continue
            # # 渲染
            # if FOR_TESTSET == 4:
            #     continue
            # if model_select == 9:
            #     if (int(epoch) + 1) % 5 != 0:
            #         continue
            png_num = predict_ind.shape[0]
            for png_i in range(png_num):
                png_path_single = all_png_path + file_tmp[png_i] + '.png'
                if not os.path.exists(png_path_single):
                    continue
                imgGt = cv2.imread(all_png_path + file_tmp[png_i] + '.png')
                # if model_select == 7:
                #     imgGt_tm2 = np.zeros([imgGt.shape[0] + 4, imgGt.shape[1] + 4, 3], dtype=np.float32)
                #     # imgData_tmp = np.zeros_like(imgData, dtype=np.float32)
                #     imgGt_tm2[2:-2, 2:-2, :] = imgGt
                #     imgGt = imgGt_tm2
                predict_ind_mask = np.zeros([imgGt.shape[0], imgGt.shape[1]], dtype=np.uint8)
                if model_select in [1, 5, 6, 8, 9, 10, 11, 12, 13, 14]:
                    # if file_tmp[png_i] in extraTest18:
                    tmp_cut_num = cut_num + int(patchsize / 2)
                    predict_ind_mask[tmp_cut_num:-tmp_cut_num, tmp_cut_num:-tmp_cut_num] = predict_ind[png_i]
                elif model_select in [3, 4]:
                    predict_ind_mask[cut_num:-cut_num, cut_num:-cut_num] = predict_ind[png_i]
                elif model_select in [7]:
                    predict_ind_mask = predict_ind[png_i][2:-2, 2:-2]
                else:
                    predict_ind_mask = predict_ind[png_i]

                predict_ind_dilate_open = None
                if dilate_open_used:
                    predict_ind_dilate_open = predict_ind_mask[15:-15, 15:-15].copy()
                    predict_ind_dilate_open = dilate_open(predict_ind_dilate_open)
                    predict_dilate_open_list.extend(predict_ind_dilate_open)
                    # print(predict_ind_dilate_open.shape)
                    # print('labelData', labelData[png_i].shape)
                    count_right_dilate_open += np.sum(predict_ind_dilate_open == labelData[png_i])  # 全部训练

                imgRes = imgGt
                for color_num in range(1, class_nums):
                    imgRes = mask_color_img(imgRes, mask=(predict_ind_mask == color_num),
                                            color=color_class[color_num - 1], alpha=0.7)
                cv2.imwrite(result_dir + file_tmp[png_i] + '.png', imgRes)
                # 保存预测结果图像
                cv2.imwrite(result_label_dir + file_tmp[png_i] + '.png', predict_ind_mask)
        # gc.collect()
    accuracy = count_right / count_tot
    accuracy_dilate = count_right_dilate_open / count_tot
    print('epoch ' + epoch + ' accuracy: ', accuracy)
    print('after dilate open accuracy: ', accuracy_dilate)

    label_list = np.array(label_list).flatten()
    predict_list = np.array(predict_list).flatten()
    ind = (label_list != 255)  # 一定要加这个
    predict_list = predict_list[ind]
    label_list = label_list[ind]
    if dilate_open_used:
        predict_dilate_open_list = np.array(predict_dilate_open_list).flatten()
        predict_dilate_open_list = predict_dilate_open_list[ind]

    target_names = ['other', 'skin_', 'cloth', 'water']
    # print(set(list(label_list)))
    # print(set(list(predict_list)))
    res = classification_report(label_list, predict_list, target_names=target_names, output_dict=True)
    res_dilate = None
    kappa_score_dilate = None
    if dilate_open_used:
        res_dilate = classification_report(label_list, predict_dilate_open_list, target_names=target_names,
                                           output_dict=True)
        kappa_score_dilate = cohen_kappa_score(label_list, predict_dilate_open_list)
    kappa_score = cohen_kappa_score(label_list, predict_list)  # 计算卡帕系数
    # accuracy = count_right / count_tot
    # print('epoch ' + epoch + ' accuracy: ',accuracy)

    csv2_line = [epoch, res['accuracy'], res['macro avg']['f1-score'], kappa_score]
    # print('accuracy=', accuracy, 'micro=', micro, 'macro', macro)
    for k in target_names:
        # print(k,'skin pre=', res['skin']['precision'], 'skin rec=', res['skin']['recall'])
        print(k, ' pre=', res[k]['precision'], ' rec=', res[k]['recall'], ' f1-score=', res[k]['f1-score'])
        csv2_line.extend([res[k]['precision'], res[k]['recall'], res[k]['f1-score']])
    print(epoch, 'all test micro accuracy:', res['accuracy'], 'macro avg :', res['macro avg']['f1-score'],
          'kappa_score: ', kappa_score)
    print('\n')
    f2_csv.writerow(csv2_line)
    if dilate_open_used:
        print('after dilate open:')
        csv2_line = [epoch, res_dilate['accuracy'], res_dilate['macro avg']['f1-score'], kappa_score_dilate]
        # print('accuracy=', accuracy, 'micro=', micro, 'macro', macro)
        for k in target_names:
            # print(k,'skin pre=', res['skin']['precision'], 'skin rec=', res['skin']['recall'])
            print(k, ' pre=', res_dilate[k]['precision'], ' rec=', res_dilate[k]['recall'], ' f1-score=', res_dilate[k]['f1-score'])
            csv2_line.extend([res_dilate[k]['precision'], res_dilate[k]['recall'], res_dilate[k]['f1-score']])
        print(epoch, 'all test micro accuracy:', res_dilate['accuracy'], 'macro avg :', res_dilate['macro avg']['f1-score'],
              'kappa_score: ', kappa_score_dilate)
        print('\n')
        f2_csv.writerow(csv2_line)
f2.close()
# csv2_header = ['epoch', 'micro accuracy','macro avg f1','other_pre','other_rec','other_f1',
#                'skin_pre', 'skin_rec','skin_f1','cloth_pre', 'cloth_rec','cloth_f1',
#                'plant_pre','plant_rec','plant_f1']
