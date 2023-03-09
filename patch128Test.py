from model_block.materialNet import *
import os
from utils.os_helper import mkdir
from utils.parse_args import parse_args
import sys
from torch.autograd import Variable
from sklearn.metrics import classification_report, f1_score
import csv

select_bands = [1, 13, 25, 52, 76, 92, 99, 105, 109]
bands_num = 9
mBatchSize = 1024
if __name__ == '__main__':
    # 模型复现
    seed = 2021
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    # 可以考虑改变benchmark为true
    torch.backends.cudnn.benchmark = False
    # 配合随机数种子，确保网络多次训练参数一致
    torch.backends.cudnn.deterministic = True
    # 使用非确定性算法
    torch.backends.cudnn.enabled = True

    save_testData_npy_path = '/home/cjl/ywj_code/graduationCode/alien_material/trainData/128HZRiverSkinClothTest_1_13_25_52_76_92_99_105_109True_False.npy'
    save_testLabel_npy_path = '/home/cjl/ywj_code/graduationCode/alien_material/trainData/128HZRiverSkinClothTest_1_13_25_52_76_92_99_105_109True_False_label.npy'

    testData = np.load(save_testData_npy_path)
    testLabel = np.load(save_testData_npy_path[:-4] + '_label.npy')

    print("HZRiver testData shape : ", testData.shape)
    print("HZRiver testLabel shape : ", testLabel.shape)
    for i in range(class_nums):
        print('HZRiver test class :', i, 'nums:', np.sum(np.nanargmax(testLabel, axis=1) == i))

    ShAndWaterPath = '/home/cjl/ywj_code/graduationCode/BS-NETs/trainData/128bandsFalse_60_mulprocess.npy'
    ShAndWaterLabelPath = '/home/cjl/ywj_code/graduationCode/BS-NETs/trainData/128bandsFalse_60_mulprocess_label.npy'
    ShAndWaterTrainData = np.load(ShAndWaterPath)  # b c h w

    ShAndWaterTrainData = ShAndWaterTrainData[:, select_bands, :, :]
    ShAndWaterTrainLabel = np.load(ShAndWaterLabelPath)  # b class_nums
    ShAndWaterTrainNums = list(range(ShAndWaterTrainData.shape[0]))
    ShAndWaterTrainPos = random.sample(ShAndWaterTrainNums, int(ShAndWaterTrainData.shape[0] * 0.7))
    ShAndWaterTestPos = [i for i in ShAndWaterTrainNums if i not in ShAndWaterTrainPos]
    # exShAndWaterTrainData = ShAndWaterTrainData[ShAndWaterTrainPos]
    exShAndWaterTestData = ShAndWaterTrainData[ShAndWaterTestPos]
    # exShAndWaterTrainLabel = ShAndWaterTrainLabel[ShAndWaterTrainPos]
    exShAndWaterTestLabel = ShAndWaterTrainLabel[ShAndWaterTestPos]

    testData = np.concatenate([testData, exShAndWaterTestData], axis=0)
    testLabel = np.concatenate([testLabel, exShAndWaterTestLabel], axis=0)
    # 打印一下每个类别的标签数量
    # print("All trainData shape : ", trainData.shape)
    # print("All trainLabel shape : ", trainLabel.shape)
    # for i in range(class_nums):
    #     print('All train class :', i, 'nums:', np.sum(np.nanargmax(trainLabel, axis=1) == i))

    print("All testData shape : ", testData.shape)
    print("All testLabel shape : ", testLabel.shape)
    for i in range(class_nums):
        print('All test class :', i, 'nums:', np.sum(np.nanargmax(testLabel, axis=1) == i))

    # sys.exit()
    # trainDataset = MyDataset(trainData, trainLabel)
    testDataset = MyDataset(testData, testLabel)
    # 训练一次 用到多少张图片（patch 像素块）32 64 128
    # trainLoader = DataLoader(dataset=trainDataset, batch_size=mBatchSize, shuffle=True)
    testLoader = DataLoader(dataset=testDataset, batch_size=mBatchSize, shuffle=True)
    model = MaterialSubModel(bands_num, class_nums).cuda()

    # criterion=nn.MSELoss()
    # 损失函数 cross交叉熵
    # criterion = nn.SmoothL1Loss()
    # 可以换一个损失函数试一下 二分类
    criterion = nn.CrossEntropyLoss()

    epoch_list = [str(x) for x in [295, 298, 282, 270, 190, 227, 277, 168, 268,
                                   163, 200, 145, 253, 234, 247, 176, 177, 158, 208, 299]]
    model_path = "/home/cjl/ywj_code/graduationCode/alien_material/model/HZWaterSkinAddSh_64_0.001_True_False_False/"
    csv2_save_path = './log/'
    mkdir(csv2_save_path)
    f2 = open(csv2_save_path + '128HZRiverSkinClothTestPatch.csv', 'w', newline='')
    f2_csv = csv.writer(f2)
    csv2_header = ['epoch', 'micro accuracy', 'macro avg f1', 'other_pre', 'other_rec', 'other_f1',
                   'skin_pre', 'skin_rec', 'skin_f1', 'cloth_pre', 'cloth_rec', 'cloth_f1',
                   'water_pre', 'water_rec', 'water_f1']
    # csv2_header = ['micro accuracy']
    f2_csv.writerow(csv2_header)

    for epoch in epoch_list:
        model.load_state_dict(torch.load(model_path + epoch + '.pkl'))
        # 一定要加测试模式 有BN层或者dropout 的都需要，最好一直有
        model.eval()
        testLossTotal = 0.0
        testCorrect = 0
        testTotal = 0
        label_list = []
        predict_list = []
        for i, data in enumerate(testLoader, 0):
            img, label = data
            img, label = Variable(img).float().cuda(), Variable(label).float().cuda()
            # img, label = torch.tensor(img).float().cuda(), torch.tensor(label).float().cuda()
            testTotal += label.size(0)
            img = img / torch.max(img, dim=1, keepdim=True)[0]  # 预处理部分不用归一化了
            with torch.no_grad():
                predict = model(img)
                predict = predict.squeeze()
                predictIndex = torch.argmax(predict, dim=1)
                # predictIndex = torch.sigmoid(predict) >= 0.5
                labelIndex = torch.argmax(label, dim=1)
                # labelIndexBool = labelIndex==1
                testCorrect += (predictIndex == labelIndex).sum()
                # loss = criterion(predict, labelIndex)
                # if torch.isnan(loss):
                #     print('is nan!!')
                # testLossTotal += loss
                predict_list.extend(predictIndex.cpu().detach().numpy())
                label_list.extend(labelIndex.cpu().detach().numpy())
        print('test epoch acc:', epoch, ': ', testCorrect.item() / testTotal)
        # print('total loss = %.5f' % float(testLossTotal))
        print('\n')
        label_list = np.array(label_list).flatten()
        predict_list = np.array(predict_list).flatten()
        ind = (label_list != 255)  # 一定要加这个
        predict_list = predict_list[ind]
        label_list = label_list[ind]
        # micro = f1_score(label_list, predict_list, average="micro")
        # macro = f1_score(label_list, predict_list, average="macro")

        target_names = ['other', 'skin_', 'cloth', 'water']
        res = classification_report(label_list, predict_list, target_names=target_names, output_dict=True)
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

