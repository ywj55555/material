import numpy as np
from waterModelBig import CNNCTC
from materialNet import *
import os
from utils.os_helper import mkdir
from utils.parse_args import parse_args
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
# CUDA:0
args = parse_args()
mBatchSize = args.batchsize
mEpochs = args.epoch
model_select = args.model_select
# mLearningRate = args.lr
mLearningRate = 0.001
mDevice=torch.device("cuda")
nora = args.nora
print('mBatchSize',mBatchSize)
print('mEpochs',mEpochs)
print('mLearningRate',mLearningRate)
print('model_select',model_select)
print('nora',nora)
model_path = ['./oriSampleWaterModel/','./newSampleWaterModel/']
model_save = model_path[model_select-1]
dataTypelist = ['water', 'water']
dataType = dataTypelist[model_select-1]
mkdir(model_save)
class_nums = 2
bands = 22

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
    trainData, trainLabel = generateData(dataType, 1000, 16, DATA_TYPE,nora=nora, class_nums = class_nums)

    # testData, testLabel = generateData(dataType, 600, 11, DATA_TYPE,nora=nora, class_nums=class_nums)
    # trainData = np.load('./trainData/train.npy')
    # trainLabel = np.load('./trainData/trainLabel.npy')
    # try:
    #     np.save('./trainData/train.npy',trainData)
    #     np.save('./trainData/trainLabel.npy', trainLabel)
    #     # np.save('./testData/testData.npy',testData)
    #     # np.save('./testData/testLabel.npy',testLabel)
    # except:
    #     print("error")
    #     pass
    trainDataset = MyDataset(trainData, trainLabel)
    # testDataset = MyDataset(testData, testLabel)

    trainLoader = DataLoader(dataset=trainDataset, batch_size=mBatchSize, shuffle=True)
    # testLoader = DataLoader(dataset=testDataset, batch_size=mBatchSize, shuffle=True)

    model = MaterialSubModel(bands, class_nums ).cuda()

    # criterion=nn.MSELoss()
    # 损失函数 cross交叉熵
    # criterion = nn.SmoothL1Loss()
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.BCELoss()
    # 优化器 SGD
    # optimizer = torch.optim.Adam(model.parameters(), lr=mLearningRate)
    optimizer = torch.optim.SGD(model.parameters(), lr=mLearningRate, momentum=0.9)

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
                print(img)
                print(img.shape)
                # print(loss)
                # print(predict)
                # print(labelIndex)
            trainLossTotal += loss
            # print("loss = %.5f" % float(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
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
