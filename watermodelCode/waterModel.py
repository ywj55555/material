import numpy as np

from materialNet import *
import os
from utils.os_helper import mkdir
from utils.parse_args import parse_args
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
# CUDA:0
args = parse_args()
# batchsize 也可以改 目前为32
mBatchSize = args.batchsize

mEpochs = args.epoch
model_select = args.model_select
# mLearningRate = args.lr
# 梯度更新步长 0.0001
mLearningRate = 0.001
mDevice=torch.device("cuda")
nora = args.nora
print('mBatchSize',mBatchSize)
print('mEpochs',mEpochs)
print('mLearningRate',mLearningRate)
print('model_select',model_select)
print('nora',nora)
# model_path = ['./IntervalSampleWaterModel/','./newSampleWaterModel/','./IntervalSampleAddFeatureWaterModel/']
# model_save = model_path[model_select-1]
model_save = './IntervalSampleAddFeatureWaterModel_shenzhen1/'
# dataTypelist = ['water', 'water','water']
# dataType = dataTypelist[model_select-1]
dataType = 'sea'

featureTrans = True#False#
mkdir(model_save)
class_nums = 2

if featureTrans:
    bands = 21
else:
    bands = 11

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
    trainData, trainLabel = generateData(dataType, 2500, 11, DATA_TYPE, nora=nora, class_nums=class_nums, intervalSelect = True, featureTrans = featureTrans)
    # trainData, trainLabel = generateData(dataType, 1000, 11, DATA_TYPE,nora=nora, class_nums = class_nums)

    # testData, testLabel = generateData(dataType, 600, 11, DATA_TYPE,nora=nora, class_nums=class_nums)
    # trainData = np.load('./trainData/train.npy')
    # trainLabel = np.load('./trainData/trainLabel.npy')
    # trainData = np.load('./trainData/trainIntervalAddFeature_1.npy')
    # trainLabel = np.load('./trainData/trainLabelIntervalAddFeature_1.npy')
    # try:
    #     np.save('./trainData/trainIntervalAddFeature_1.npy',trainData)
    #     np.save('./trainData/trainLabelIntervalAddFeature_1.npy', trainLabel)
    #     # np.save('./testData/testData.npy',testData)
    #     # np.save('./testData/testLabel.npy',testLabel)
    # except:
    #     print("error")
    #     pass

    print("begin!!!")
    trainDataset = MyDataset(trainData, trainLabel)
    # testDataset = MyDataset(testData, testLabel)
    # 训练一次 用到多少张图片（patch 像素块）32 64 128
    trainLoader = DataLoader(dataset=trainDataset, batch_size=mBatchSize, shuffle=True)
    # testLoader = DataLoader(dataset=testDataset, batch_size=mBatchSize, shuffle=True)

    model = MaterialSubModel(bands, class_nums ).cuda()

    # criterion=nn.MSELoss()
    # 损失函数 cross交叉熵
    # criterion = nn.SmoothL1Loss()
    # 可以换一个损失函数试一下 二分类
    criterion = nn.CrossEntropyLoss()

    # criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.BCELoss()
    # 优化器 SGD ADAM 可以换一个优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=mLearningRate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=mLearningRate, momentum=0.9)
    # optimizer = torch.optim.Adam(parameters, lr=mLearningRate, weight_decay=1e-4)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=mLearningRate)

    # 可以补充一个 学习率 随着训练 epoch 或者 准确率 变化的 配置
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8, last_epoch=-1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15,T_mult=2)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 15, eta_min=0.0, last_epoch=-1)
    # lr_adjust = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 15, eta_min=0.0, last_epoch=-1)
    # lr_adjust = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 15, eta_min=0.0, last_epoch=-1)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=8,
    #                                                        verbose=True, threshold=0.005, threshold_mode='rel',
    #                                                        cooldown=0,
    #                                                        min_lr=0, eps=1e-08)

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
            optimizer.zero_grad() # 清空上一次梯度信息
            loss.backward() # 计算梯度
            optimizer.step()  # 更新参数，用到学习率 学习率越大 更新步长越大
        print('train epoch:', epoch, ': ', trainCorrect.item() / trainTotal)
        print('total loss = %.5f' % float(trainLossTotal))
        # torch.save(model.state_dict(), "model/concat11/params" + str(epoch) + ".pkl")
        # 测试
        # testCorrect = 0
        # testTotal = 0
        # for i, data in enumerate(testLoader, 0):
        #     img, label = data
        #     # img, label = torch.tensor(img).float().cuda(), torch.tensor(label).float().cuda()
        #     img, label = Variable(img).float().cuda(), Variable(label).float().cuda()
        #     # tmp = torch.Tensor(img).to(mDevice)
        #
        #     testTotal += label.size(0)
        #     # 测试阶段不需要保存梯度信息
        #     with torch.no_grad():
        #         predict = model(img)
        #
        #         predict = torch.squeeze(predict)
        #         # 计算正确率
        #         predictIndex = torch.argmax(predict, dim=1)
        #         labelIndex = torch.argmax(label, dim=1)
        #         testCorrect += (predictIndex == labelIndex).sum()
        # print('test epoch:', epoch, ': ', testCorrect.item() / testTotal)
        # print('\n')
        # # if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), model_save + str(epoch) + '.pkl')
