from torch.autograd import Variable
from model_block import network
from model_block.materialNet import *
import os
from utils.os_helper import mkdir
from utils.parse_args import parse_args
# from model_block.spaceSpectrumFusionNet import spaceSpectrumFusionNet
import sys

# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
# CUDA:0
args = parse_args()
mBatchSize = args.batchsize
mEpochs = args.epoch
# mLearningRate = 0.001
mLearningRate = args.lr
mDevice = torch.device("cuda")
model_select = args.model_select
if model_select in [3, 5]:
    mBatchSize = mBatchSize * 16 * 4 * 2
# class_num = args.class_num

# 输入为 ”“ 代表为false 其他则为true
# nora = args.nora
# allband = args.allband
# allband = False
# feature = args.feature
feature = False
intervalSelect = True
# hand_selected_bands = [1, 13, 25, 52, 76, 92, 99, 105, 109]
# 对应到18通道数据
# 0     1   2   3   4   5   6  7   8   9   10   11  12  13  14  15  16  17
# {453,467,483,495,512,528,539,555,569,620,638,657,683,711,730,755,779,816}
# 453nm、467nm、495nm、555nm、638nm、711nm、755nm、779nm、816nm
hand_selected_bands = [0, 1, 3, 7, 10, 13, 15, 16, 17]
# embedding_selected_bands = [17,  8, 11, 16,  1, 12,  9, 10, 14] # 17  8 11 16  1  9 10 12 14
# embedding_selected_bands.sort()
input_bands_num = len(hand_selected_bands)
bands_sum = 1
class_num = 4  # 0 other 1 skin 2 cloth 3 water
for i in hand_selected_bands:
    if i == 0:
        bands_sum = bands_sum + 1
        continue
    bands_sum = bands_sum * i
model_list = ['smallMolde', 'DBDA', 'SSRN', 'DBMA', 'FDSSC']

model_save = './model/SkinClothWater18_' + model_list[model_select - 1] + '_' + str(mLearningRate) + '_' + str(mBatchSize) + '_' + str(
    class_num) + '_handSelect_' + str(bands_sum) + '/'


# if class_num == 4:
#     per_class_num = [500, 500, 1500]
# elif class_num == 10:
#     per_class_num = [300 for _ in range(10)]
print('model_select', model_select)
print('batch', mBatchSize)
print('epoch', mEpochs)
print('lr', mLearningRate)
print('class_num', class_num)
nora = False
if nora:
    print("nora yes")
else:
    print("nora no")
allband = False

print("allband:", allband)
print('feature:', feature)

mkdir(model_save)


if __name__ == '__main__':
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

    # trainData, trainLabel = generateData('train', 300, 11, DATA_TYPE, nora=nora, allband=allband, feature=feature)
    # testData, testLabel = generateData('test', 300, 11, DATA_TYPE, nora=nora, allband=allband, feature=feature)
    trainDataType = 'allTrain18'
    testDataType = 'allTest18'
    bands_str = '18'
    addSpace = True
    save_npy_path = './trainData/'
    save_trainData_npy_path = save_npy_path + bands_str + str(intervalSelect) + '_' + str(class_num) + 'class.npy'
    mkdir(save_npy_path)
    # save_trainData_npy_path = './trainData/big_32_0.001_Falsemulprocess1.npy'
    print(save_trainData_npy_path)
    if not os.path.exists(save_trainData_npy_path):
        # trainData, trainLabel = generateData('train', 300, 11 , nora=nora, allband=allband, feature=feature,class_num=class_num,cut_num=10,per_class_num = per_class_num)
        # testData, testLabel = generateData('test', 300, 11 , nora=nora, allband=allband, feature=feature,class_num=class_num,cut_num=10,per_class_num=per_class_num)
        # 再加上切面的patch数据！！
        trainData, trainLabel, trainDataSpace = multiProcessGenerateData(trainDataType, 2000, 11, nora=nora,
                                                         class_nums=class_num,
                                                         intervalSelect=intervalSelect, addSpace=addSpace)
        testData, testLabel, testDataSpace = multiProcessGenerateData(testDataType, 2000, 11,
                                                       nora=nora,
                                                       class_nums=class_num,
                                                       intervalSelect=intervalSelect,addSpace=addSpace)

        try:
            np.save(save_trainData_npy_path, trainData)
            np.save(save_trainData_npy_path[:-4] + '_label.npy', trainLabel)
            np.save(save_trainData_npy_path[:-4] + '_space.npy', trainDataSpace)

            np.save(save_trainData_npy_path[:-4] + '_test.npy', testData)
            np.save(save_trainData_npy_path[:-4] + '_testLabel.npy', testLabel)
            np.save(save_trainData_npy_path[:-4] + '_testSpace.npy', testDataSpace)

        except:
            pass
    else:
        trainData = np.load(save_trainData_npy_path)
        trainLabel = np.load(save_trainData_npy_path[:-4] + '_label.npy')
        # trainDataSpace = np.load(save_trainData_npy_path[:-4] + '_space.npy')
        testData = np.load(save_trainData_npy_path[:-4] + '_test.npy')
        testLabel = np.load(save_trainData_npy_path[:-4] + '_testLabel.npy')
        # testDataSpace = np.load(save_trainData_npy_path[:-4] + '_testSpace.npy')

    trainData = trainData[:, hand_selected_bands, :, :]
    testData = testData[:, hand_selected_bands, :, :]
    trainData = trainData / np.max(trainData, axis=1, keepdims=True)  # 通道归一化
    testData = testData / np.max(testData, axis=1, keepdims=True)  # 通道归一化

    print('testData shape : ', testData.shape)
    print('testLabel shape : ', testLabel.shape)
    print('trainData shape : ', trainData.shape)
    print('trainLabel shape : ', trainLabel.shape)
    print('Number of training sets in each category')
    for i in range(class_num):
        print('class :', i, 'nums:', np.sum(np.nanargmax(trainLabel, axis=1) == i))
    print('Number of test sets in each category')
    for i in range(class_num):
        print('class :', i, 'nums:', np.sum(np.nanargmax(testLabel, axis=1) == i))

    trainDataset = MyDataset(trainData, trainLabel)
    testDataset = MyDataset(testData, testLabel)
    trainLoader = DataLoader(dataset=trainDataset, batch_size=mBatchSize, shuffle=True)
    testLoader = DataLoader(dataset=testDataset, batch_size=mBatchSize, shuffle=True)
    if model_select == 1:
        model = MaterialSubModel(input_bands_num, class_num).cuda()
    elif model_select == 2:
        model = network.DBDA_network_MISH(input_bands_num, class_num).cuda()
    elif model_select == 3:  # 减少参数量！！！
        model = network.SSRN_network(input_bands_num, class_num).cuda()
    elif model_select == 4:
        model = network.DBMA_network(input_bands_num, class_num).cuda()
    elif model_select == 5:  # 减少参数量！！！
        model = network.FDSSC_network(input_bands_num, class_num).cuda()
    # elif model_select == 6:
    #     model = spaceSpectrumFusionNet(input_bands_num, class_num).cuda()
    else:
        model = MaterialSubModel(input_bands_num, class_num).cuda()
    # criterion = nn.SmoothL1Loss()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=mLearningRate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=8,
                                                           verbose=True, threshold=0.005, threshold_mode='rel',
                                                           cooldown=0,
                                                           min_lr=0, eps=1e-08)
    # lr_adjust = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 15, eta_min=0.0, last_epoch=-1)
    for epoch in range(mEpochs):
        # 训练
        trainCorrect = 0
        trainTotal = 0
        trainLossTotal = 0.0
        for i, data in enumerate(trainLoader, 0):
            img, label = data
            img, label = Variable(img).float().cuda(), Variable(label).float().cuda()
            # img, label = torch.tensor(img).float().cuda(), torch.tensor(label).float().cuda()
            trainTotal += label.size(0)
            # 高光谱 3D 卷积模型
            if model_select != 1:
                img = img.permute(0, 2, 3, 1)
                img = img.unsqueeze(1)  # 增加通道维度
            predict = model(img)
            # if model_select == 3:
            #     print(predict.size())
            # 计算正确率 B C 1 1
            predict = predict.squeeze()
            predictIndex = torch.argmax(predict, dim=1)
            labelIndex = torch.argmax(label, dim=1)
            trainCorrect += (predictIndex == labelIndex).sum()
            # 产生loss
            # loss = criterion(predict, label)
            # nn.CrossEntropyLoss() 需要使用labelIndex计算loss
            loss = criterion(predict, labelIndex)
            trainLossTotal += loss
            # print("loss = %.5f" % float(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('train epoch:', epoch, ': ', trainCorrect.item() / trainTotal)
        print('total loss = %.5f' % float(trainLossTotal))

        # torch.save(model.state_dict(), "model/concat11/params" + str(epoch) + ".pkl")
        # 测试
        testCorrect = 0
        testTotal = 0
        testLossTotal = 0
        for i, data in enumerate(testLoader, 0):
            img, label = data
            # img, label = torch.tensor(img).float().cuda(), torch.tensor(label).float().cuda()
            img, label = Variable(img).float().cuda(), Variable(label).float().cuda()
            # tmp = torch.Tensor(img).to(mDevice)
            testTotal += label.size(0)
            with torch.no_grad():
                if model_select != 1:
                    img = img.permute(0, 2, 3, 1)
                    img = img.unsqueeze(1)  # 增加通道维度
                predict = model(img)
                predict = torch.squeeze(predict)
                # 计算正确率
                predictIndex = torch.argmax(predict, dim=1)
                labelIndex = torch.argmax(label, dim=1)
                testCorrect += (predictIndex == labelIndex).sum()
                # testloss = criterion(predict, label)
                # nn.CrossEntropyLoss() 需要使用labelIndex计算loss
                testloss = criterion(predict, labelIndex)
                testLossTotal += testloss

        print('test epoch:', epoch, ': ', testCorrect.item() / testTotal)
        print('test loss:', epoch, ':', testLossTotal.item())
        print('\n')
        # if (epoch + 1) % 5 == 0:
        # if mLearningRate != 0.0001:
        scheduler.step(testLossTotal.item())
        print('learning rate = ', optimizer.state_dict()['param_groups'][0]['lr'])

        torch.save(model.state_dict(), model_save + str(epoch) + '.pkl')
