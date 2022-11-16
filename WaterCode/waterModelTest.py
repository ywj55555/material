from materialNet import *
import cv2
import gc
from utils.add_color import mask_color_img
from utils.os_helper import *
from utils.accuracy_helper import *
from utils.parse_args import parse_test_args
import warnings
warnings.filterwarnings("ignore")
from materialNet import *
from Dataloader import *
import os
from utils.os_helper import mkdir
from torch.autograd import Variable
from model_block.PP_liteseg_final import PPLiteSeg
from tqdm import tqdm



args = parse_test_args()
FOR_TESTSET = args.FOR_TESTSET
mBatchSize = args.test_batch


select_bands = [2,36,54,61,77,82,87,91,95,104,108]
select_bands = [x + 5 for x in  select_bands]
# imgpath
train_set = '_a_HF'#'_x'#'_all_data_without_label'#'_all'#'_a'#
# png_path = '/home/glk/datasets/Water_shenzhen/RGB/'#hefei_test
png_path = '/home/glk/datasets/hefei/All_data/RGB/'#
# png_path = '/home/glk/datasets/hangzhou/rgb/'

model_select = 5#4#
class_nums = 2
dim = 13#11#12
if model_select == 1:
    model = MaterialSubModel(dim, class_nums + 1).cuda()
elif model_select == 2:
    model = CNN(dim, 2).cuda()
elif model_select == 3:
    from DBDA_net import DBDA_network_with_2D

    model = DBDA_network_with_2D(dim, class_nums + 1).cuda()
elif model_select == 4:
    from DBDA_net import DBDA_network_without_attention
    model = DBDA_network_without_attention(dim, class_nums + 1).cuda()
elif model_select == 5:
    from DBDA_net import DBDA_network_three_losses

    model = DBDA_network_three_losses(dim, class_nums + 1).cuda()
    gama = 0.1
elif model_select == 6:
    from DBDA_net import DBDA_network_three_losses_cross_Attention

    model = DBDA_network_three_losses_cross_Attention(dim, class_nums + 1).cuda()#
    gama = 0.1
elif model_select == 7:
    model = PPLiteSeg(class_nums + 1, dim)
elif model_select == 8:
    from DBDA_net import DBDA_with_only_spatial

    model = DBDA_with_only_spatial(dim, class_nums + 1).cuda()

color_class = [[0,0,255],[255,0,0],[0,255,0]]
model_path = "/home/glk/watermodelCode/WHOLE_model_select_5min_kept_500000_1_0.001_False_False_sig/dim_13_x_F/299_test_acc_0.9542836131583073_spec_acc_0.9511361158547894_spat_acc_0.9467946990643376.pkl"
featureTrans = False

if __name__ == '__main__':
    model.load_state_dict(torch.load(model_path))
    dataset = 'test'
    testDataset = MyDataset_whole(train_set, dim=dim, feature_extraction=featureTrans, dataType=dataset)
    testLoader = DataLoader(dataset=testDataset, batch_size=1, shuffle=False, num_workers=10, pin_memory=False)
    testCorrect = 0
    testCorrect_spec = 0
    testCorrect_spat = 0
    testTotal = 0
    if dataset == 'test':
        # data_file = "/home/glk/datasets/Water_shenzhen/test" + train_set + ".txt"
        # data_file = "/home/glk/datasets/hefei/hefei_test/test" + train_set + ".txt"#Water_shenzhen#hefei_test2
        # data_file = "/home/glk/datasets/hangzhou/test" + train_set + ".txt"#Water_shenzhen
        data_file = "/home/glk/datasets/hefei/All_data/test" + train_set + ".txt"
    else:
        data_file = "/home/glk/datasets/Water_shenzhen/train" + train_set + ".txt"
    with open(data_file, 'r') as f:
        dataFile = f.readlines()
    # model.eval()
    result_dir = './resTrain/'
    mkdir(result_dir)
    for i, data in enumerate(tqdm(testLoader)):
        img, label = data
        # img, label = torch.tensor(img).float(), torch.tensor(label).long()#.cuda().cuda()
        img, label = Variable(img).float().cuda(), Variable(label).long().cuda()##.cuda()
        # tmp = torch.Tensor(img).to(mDevice)
        with torch.no_grad():
            if model_select == 5 or model_select == 6:
                predict, predict_spec, predict_spat = model(img)
                predictIndex_spec = torch.argmax(predict_spec, dim=1)
                # testCorrect_spec += ((predictIndex_spec == label) & (label != 0)).sum()  ##
                predictIndex_spat = torch.argmax(predict_spat, dim=1)
                # testCorrect_spat += ((predictIndex_spat == label) & (label != 0)).sum()
            else:
                predict = model(img)
        # 计算正确率

        predict = predict#predict_spat#
        # predict = predict.softmax(dim=1)
        # predict[:,0] = 0.8
        predictIndex = torch.argmax(predict, dim=1)
        # testCorrect += ((predictIndex == label) & (label != 0)).sum()
        # testTotal += torch.sum(label != 0).item()

        ##########################Paint!##############################
        predict_ind = predictIndex.cpu().detach().numpy()
        png_path_single = png_path + dataFile[i].split('\n')[0] + '.png'
        if not os.path.exists(png_path_single):
            png_path_single = png_path + dataFile[i].split('\n')[0] + '.tif'
        # print(png_path_single)
        imgGt = cv2.imread(png_path_single)
        # imgGt = imgGt[5:-5,5:-5]
        # print(imgGt.shape)
        imgRes = imgGt
        color_num = 1
        try:
            imgRes = mask_color_img(imgRes, mask=(predict_ind[0] == color_num), color=color_class[color_num - 1], alpha=0.6)
        except:
            print(png_path_single)
            continue
        # print(result_dir + dataFile[i])
        cv2.imwrite(result_dir + dataFile[i].split('\n')[0] + '.png', imgRes)
        gc.collect()


    # print('test acc:', testCorrect.item() / testTotal)
    # if model_select == 5 or model_select == 6:
    #     print('spec_acc: ', testCorrect_spec / testTotal)  # .item()
    #     print('spat_acc: ', testCorrect_spat / testTotal)  # .item()
    # print('\n')




