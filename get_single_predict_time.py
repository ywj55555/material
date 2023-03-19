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
import sys

warnings.filterwarnings("ignore")
import csv

args = parse_test_args()
FOR_TESTSET = args.FOR_TESTSET
# FOR_TESTSET = 1
test_batch = args.test_batch
# test_batch = 4
model_select = args.model_select
# model_select = 1
print(FOR_TESTSET, test_batch)
print(model_select)
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
test_file_type = ['train', 'test', 'extraTest']
color_class = [[0, 0, 255], [255, 0, 0], [0, 255, 0]]
mean = torch.tensor([0.5, 0.5, 0.5]).cuda()
std = torch.tensor([0.5, 0.5, 0.5]).cuda()
print(inputBands)
patchsize = 11
# model_name = {9:'twoBranch', 2:'PPLiteSeg', 3:'FreeNet', 4:'SSDGL', 6:'MaterialSubModel',
#               7:'BiSeNetV2'}
for model_select in [7, 7, 2, 3, 4, 6, 9, ]:  # [9, 2, 3, 4, 6, 7]
    inputData = None
    tmp_model_path = model_root_path + model_path[model_select - 1] + '/' + '4' + '.pkl'
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
        inputData = torch.rand(1, 9, 1020, 1020).cuda()
        inputSpaceData = torch.rand(1, 9, 1020, 1020).cuda()
    elif model_select == 14:  # SpectrumNet
        model = SpectrumNet(inputBands, class_nums, patch_size=11, version=3).cuda()
        inputData = torch.rand(1, 9, 1020, 1020).cuda()
    elif model_select == 2:  # PPLiteSeg
        cut_num = 0
        inputData = torch.rand(1, 3, 1020, 1020).cuda()
        model = PPLiteSeg(num_classes=class_nums, input_channel=3).cuda()
    elif model_select == 3:  # FreeNet
        cut_num = 6
        model = FreeNet(bands=inputBands, class_nums=class_nums).cuda()
        inputData = torch.rand(1, 9, 1008, 1008).cuda()
    elif model_select == 4:  # SSDGL
        cut_num = 6
        model = SSDGL(bands=inputBands, class_nums=class_nums).cuda()
        inputData = torch.rand(1, 9, 1008, 1008).cuda()
    elif model_select == 6:  # smallModel
        model = MaterialSubModel(inputBands, class_nums).cuda()
        inputData = torch.rand(1, 9, 1020, 1020).cuda()
    elif model_select == 7:  # BisNetv2
        cut_num = 0
        model = BiSeNetV2(n_classes=class_nums, aux_mode='train').cuda()
        inputData = torch.rand(1, 3, 1024, 1024).cuda()
        model.aux_mode = 'eval'
    else:
        print('model none!!')
        model = None
        sys.exit()
    model.load_state_dict(torch.load(tmp_model_path))
    # 一定要加测试模式 有BN层或者dropout 的都需要，最好一直有
    model.eval()
    with torch.no_grad():
        torch.cuda.empty_cache()
        cnt = 10
        start = time.time()
        for i in range(cnt):
            if model_select in [1, 5, 8, 9, 10, 11, 12, 13]:
                predict = model(inputData, inputSpaceData)
            else:
                predict = model(inputData)
            # if model_select != 2:
            #     predict_ind = torch.argmax(predict, dim=1)
            # else:
            #     predict_ind = torch.argmax(predict[0], dim=1)
        end = time.time()
        print('model:', model_select, model_name[model_select - 1], 'cost average :', (end - start) / cnt)
