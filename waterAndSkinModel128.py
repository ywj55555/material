from model_block.materialNet import *
import os
from utils.os_helper import mkdir
from utils.parse_args import parse_args
import sys
from torch.autograd import Variable

# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
# CUDA:0
args = parse_args()
# batchsize 也可以改 目前为32
mBatchSize = args.batchsize

mEpochs = args.epoch
model_select = args.model_select
mLearningRate = args.lr
# 梯度更新步长 0.0001
mDevice = torch.device("cuda")
nora = args.nora  # True
nora = False
print('mBatchSize', mBatchSize)
print('mEpochs', mEpochs)
print('mLearningRate', mLearningRate)
print('model_select', model_select)
print('nora', nora)
# featureTrans = args.featureTrans  # False#
featureTrans = False
print('featureTrans', featureTrans)
# select_bands = [2,36,54,61,77,82,87,91,95,104,108]
# select_bands = [x + 5 for x in  select_bands]

# select_bands = [116, 125, 109, 100, 108,  53,  98,  90,  81, 127, 123,  19]
# select_bands = [22, 38, 57, 68, 77, 86, 90, 100, 105, 112, 115, 123]  # 皮肤、衣物 和 水 的特征波段
select_bands = [1, 13, 25, 52, 76, 92, 99, 105, 109]
# 最后一个epoch 选取结果 --》 109  98  81 116 125  90 112 127 108 118 100 123
# 116 125 109 100 108  53  98  90  81 127 123  19
class_nums = 4  # 0:other 1:skin 2:cloth 3:water
if featureTrans:
    bands_num = 21
else:
    bands_num = len(select_bands)

if bands_num != 128:
    bands_str = [str(x) for x in select_bands]
    bands_str = "_".join(bands_str)
else:
    bands_str = "128bands"

intervalSelect = args.intervalSelect  # True
print('intervalSelect :', intervalSelect)
activa = args.activa
print('activa :', activa)
from utils.load_spectral import Sigmoid
from utils.load_spectral import Tanh

if activa == 'sig':
    activate = Sigmoid
else:
    activate = Tanh

model_save = "./model/HZWaterSkinAddSh_" + str(mBatchSize) + "_" + str(mLearningRate) + "_" + str(
    intervalSelect) + "_" + str(nora) + \
             "_" + str(featureTrans) + "/"
dataType = 'RiverSkinDetectionAll'
trainDataType = 'HZRiverSkinClothTrain'
testDataType = 'HZRiverSkinClothTest'
mkdir(model_save)
# 学习率 batchsize 损失函数 优化器 增大模型
# SGD 和 Adam 看吴恩达视频

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

    # 生成训练数据 和 测试数据
    # 像素块 对应类别 列表形式

    # 换波段
    # 换清晰一点的视频
    # 改变训练标签
    # save_trainData_npy_path = './trainData/' + model_save[2:-1] + str(len(select_bands)) + '_mulprocess.npy'
    save_npy_path = './trainData/'
    mkdir(save_npy_path)
    save_trainData_npy_path = save_npy_path + '128HZRiverSkinClothTrain' + '_' + bands_str + \
                              str(intervalSelect) + "_" + str(featureTrans) + '.npy'
    save_testData_npy_path = save_npy_path + '128HZRiverSkinClothTest' + '_' + bands_str + str(
        intervalSelect) + "_" + str(featureTrans) + '.npy'

    print(save_trainData_npy_path)

    if not os.path.exists(save_trainData_npy_path):
        #     # 数据的归一化 应该在分割完patch之后 避免以后需要不归一化的数据
        # 没有归一化
        # 用两个进程？ 肯定不行啊，每个函数都是多进程
        trainData, trainLabel = multiProcessGenerateData(trainDataType, 5000, 11, select_bands, activate, nora=nora,
                                                         class_nums=class_nums,
                                                         intervalSelect=intervalSelect, featureTrans=featureTrans)
        testData, testLabel = multiProcessGenerateData(testDataType, 5000, 11, select_bands, activate, nora=nora,
                                                       class_nums=class_nums,
                                                       intervalSelect=intervalSelect, featureTrans=featureTrans)
        if not os.path.exists(save_trainData_npy_path):
            try:
                np.save(save_trainData_npy_path, trainData)
                np.save(save_trainData_npy_path[:-4] + '_label.npy', trainLabel)
                np.save(save_testData_npy_path, testData)
                np.save(save_testData_npy_path[:-4] + '_label.npy', testLabel)
            except Exception as e:
                print("error")
                print(e)
                sys.exit()
    else:
        print("train data exist!!!")
        trainData = np.load(save_trainData_npy_path)
        trainLabel = np.load(save_trainData_npy_path[:-4] + '_label.npy')
        testData = np.load(save_testData_npy_path)
        testLabel = np.load(save_testData_npy_path[:-4] + '_label.npy')
        print("load data successfully!!!")
    print("begin!!!")
    print("HZRiver trainData shape : ", trainData.shape)
    print("HZRiver trainLabel shape : ", trainLabel.shape)
    # 打印一下每个类别的标签数量
    for i in range(class_nums):
        print('HZRiver train class :', i, 'nums:', np.sum(np.nanargmax(trainLabel, axis=1) == i))

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
    exShAndWaterTrainData = ShAndWaterTrainData[ShAndWaterTrainPos]
    exShAndWaterTestData = ShAndWaterTrainData[ShAndWaterTestPos]
    exShAndWaterTrainLabel = ShAndWaterTrainLabel[ShAndWaterTrainPos]
    exShAndWaterTestLabel = ShAndWaterTrainLabel[ShAndWaterTestPos]

    trainData = np.concatenate([trainData, exShAndWaterTrainData], axis=0)
    testData = np.concatenate([testData, exShAndWaterTestData], axis=0)
    trainLabel = np.concatenate([trainLabel, exShAndWaterTrainLabel], axis=0)
    testLabel = np.concatenate([testLabel, exShAndWaterTestLabel], axis=0)
    # 打印一下每个类别的标签数量
    print("All trainData shape : ", trainData.shape)
    print("All trainLabel shape : ", trainLabel.shape)
    for i in range(class_nums):
        print('All train class :', i, 'nums:', np.sum(np.nanargmax(trainLabel, axis=1) == i))

    print("All testData shape : ", testData.shape)
    print("All testLabel shape : ", testLabel.shape)
    for i in range(class_nums):
        print('All test class :', i, 'nums:', np.sum(np.nanargmax(testLabel, axis=1) == i))

    sys.exit()
    trainDataset = MyDataset(trainData, trainLabel)
    testDataset = MyDataset(testData, testLabel)
    # 训练一次 用到多少张图片（patch 像素块）32 64 128
    trainLoader = DataLoader(dataset=trainDataset, batch_size=mBatchSize, shuffle=True)
    testLoader = DataLoader(dataset=testDataset, batch_size=mBatchSize, shuffle=True)
    model = MaterialSubModel(bands_num, class_nums).cuda()

    # criterion=nn.MSELoss()
    # 损失函数 cross交叉熵
    # criterion = nn.SmoothL1Loss()
    # 可以换一个损失函数试一下 二分类
    criterion = nn.CrossEntropyLoss()

    # criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.BCELoss()
    # 优化器 SGD ADAM 可以换一个优化器
    # optimizer = torch.optim.Adam(model.parameters(), lr=mLearningRate)
    optimizer = torch.optim.SGD(model.parameters(), lr=mLearningRate, momentum=0.9)
    # optimizer = torch.optim.Adam(parameters, lr=mLearningRate, weight_decay=1e-4)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=mLearningRate)

    # 可以补充一个 学习率 随着训练 epoch 或者 准确率 变化的 配置
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8, last_epoch=-1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15,T_mult=2)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 15, eta_min=0.0, last_epoch=-1)
    # lr_adjust = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 15, eta_min=0.0, last_epoch=-1)
    # lr_adjust = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 15, eta_min=0.0, last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=8,
                                                           verbose=True, threshold=0.005, threshold_mode='rel',
                                                           cooldown=0,
                                                           min_lr=0, eps=1e-08)

    for epoch in range(mEpochs):
        # 训练
        trainCorrect = 0
        trainTotal = 0
        trainLossTotal = 0.0
        for i, data in enumerate(trainLoader, 0):
            # 每次取出batchsize个 取出像素块  及其 对应类别
            img, label = data
            img, label = Variable(img).float().cuda(), Variable(label).float().cuda()
            # img, label = torch.tensor(img).float().cuda(), torch.tensor(label).float().cuda()
            trainTotal += label.size(0)
            img = img / torch.max(img, dim=1, keepdim=True)[0]  # 预处理部分不用归一化了
            predict = model(img)

            # 计算正确率
            # print(predict.shape)
            predict = predict.squeeze()
            predictIndex = torch.argmax(predict, dim=1)
            # predictIndex = torch.sigmoid(predict) >= 0.5
            labelIndex = torch.argmax(label, dim=1)
            # labelIndexBool = labelIndex==1
            trainCorrect += (predictIndex == labelIndex).sum()
            # 产生loss
            # labelIndex = torch.argmax(label, dim=1)
            # 交叉熵 损失函数传入 类别 而不是onehot
            # print(predict)
            # print(predict.shape)
            # print(label.shape)
            # label = label.long().squeeze()
            # predicSig = torch.sigmoid(predict)
            # labelIndex = labelIndex.float()
            # print(predicSig)
            # print(labelIndex)
            loss = criterion(predict, labelIndex)
            if torch.isnan(loss):
                print('is nan!!')
                # print(img)
                # print(img.sum())
            #     print(img.shape)
            #     print(loss)
            #     print(predict.shape)
            # print(labelIndex)
            trainLossTotal += loss
            # print("loss = %.5f" % float(loss))
            optimizer.zero_grad()  # 清空上一次梯度信息
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新参数，用到学习率 学习率越大 更新步长越大
        print('train epoch:', epoch, ': ', trainCorrect.item() / trainTotal)
        print('total loss = %.5f' % float(trainLossTotal))
        # torch.save(model.state_dict(), "model/concat11/params" + str(epoch) + ".pkl")
        # 测试
        testLossTotal = 0.0
        testCorrect = 0
        testTotal = 0
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
                loss = criterion(predict, labelIndex)
                if torch.isnan(loss):
                    print('is nan!!')
                testLossTotal += loss
        print('test epoch:', epoch, ': ', testCorrect.item() / testTotal)
        print('total loss = %.5f' % float(testLossTotal))
        print('\n')
        if mLearningRate != 0.0001:
            scheduler.step(trainLossTotal.item())
        torch.save(model.state_dict(), model_save + str(epoch) + '.pkl')
