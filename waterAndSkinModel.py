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
mDevice=torch.device("cuda")
nora = args.nora
print('mBatchSize',mBatchSize)
print('mEpochs',mEpochs)
print('mLearningRate',mLearningRate)
print('model_select',model_select)
print('nora',nora)
featureTrans = args.featureTrans#False#

print('featureTrans',featureTrans)
# select_bands = [2,36,54,61,77,82,87,91,95,104,108]
# select_bands = [x + 5 for x in  select_bands]

# select_bands = [116, 125, 109, 100, 108,  53,  98,  90,  81, 127, 123,  19]
select_bands = [22, 38, 57, 68, 77, 86, 90, 100, 105, 112, 115, 123]  # 皮肤、衣物 和 水 的特征波段
# 最后一个epoch 选取结果 --》 109  98  81 116 125  90 112 127 108 118 100 123
# 116 125 109 100 108  53  98  90  81 127 123  19
class_nums = 4
if featureTrans:
    bands_num = 21
else:
    bands_num = len(select_bands)

if bands_num != 128:
    bands_str = [str(x) for x in select_bands]
    bands_str = "_".join(bands_str)
else:
    bands_str = "128bands"

intervalSelect = args.intervalSelect
print('intervalSelect :', intervalSelect)
activa = args.activa
print('activa :', activa)
from utils.load_spectral import Sigmoid
from utils.load_spectral import Tanh
if activa == 'sig':
    activate = Sigmoid
else:
    activate = Tanh

model_save = "./model/addWaterAndSkinAll_" + str(mBatchSize) + "_" + str(mLearningRate) + "_" + str(intervalSelect) + "_" + str(nora) + \
             "_" + str(featureTrans) + "/"
dataType = 'RiverSkinDetectionAll'
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
    save_trainData_npy_path = save_npy_path + 'AddWaterClass' + dataType + '_' + bands_str + str(intervalSelect) + "_"  + str(featureTrans) + '.npy'
    # save_trainData_npy_path = './trainData/big_32_0.001_Falsemulprocess1.npy'
    print(save_trainData_npy_path)
    # multiProcessGenerateData(dataType, num, length, nora=True, class_nums=2, intervalSelect=True, featureTrans=True)
    # trainData, trainLabel = multiProcessGenerateData(dataType, 2500, 11, nora=nora, class_nums=class_nums,
    #                                      intervalSelect=True, featureTrans=featureTrans)
    # try:
    #     np.save(save_trainData_npy_path, trainData)
    #     np.save(save_trainData_npy_path[:-4] + '_label.npy', trainLabel)
    # except:
    #     print("error")
    #     sys.exit()
    # sys.exit()

    # if not os.path.exists(save_trainData_npy_path):
    #     # 数据的归一化 应该在分割完patch之后 避免以后需要不归一化的数据
    #     trainData, trainLabel = multiProcessGenerateData(dataType, 3500, 11, select_bands, activate, nora=nora, class_nums=class_nums,
    #                                                      intervalSelect=intervalSelect, featureTrans=featureTrans)
    #     # trainData, trainLabel = generateData("dataType", 2500, 11, DATA_TYPE, nora=nora, class_nums=class_nums, intervalSelect = True, featureTrans = featureTrans)
    # # trainData, trainLabel = generateData(dataType, 1000, 11, DATA_TYPE,nora=nora, class_nums = class_nums)
    #
    # # testData, testLabel = generateData(dataType, 600, 11, DATA_TYPE,nora=nora, class_nums=class_nums)
    # # trainData = np.load('./trainData/train.npy')
    # # trainLabel = np.load('./trainData/trainLabel.npy')
    # # trainData = np.load('./trainData/trainIntervalAddFeature_1.npy')
    # # trainLabel = np.load('./trainData/trainLabelIntervalAddFeature_1.npy')
    #     if not os.path.exists(save_trainData_npy_path):
    #         try:
    #             np.save(save_trainData_npy_path,trainData)
    #             np.save(save_trainData_npy_path[:-4] + '_label.npy', trainLabel)
    #             # np.save('./testData/testData.npy',testData)
    #             # np.save('./testData/testLabel.npy',testLabel)
    #         except Exception as e:
    #             print("error")
    #             print(e)
    #             sys.exit()
    # else:
    #     trainData = np.load(save_trainData_npy_path)
    #     trainLabel = np.load(save_trainData_npy_path[:-4] + '_label.npy')
    #     print("train data exist!!!")
    save_trainData_npy_path = './trainData/AddWaterClassRiverSkinDetection1_22_38_57_68_77_86_90_100_105_112_115_123True_False.npy'
    trainData1 = np.load(save_trainData_npy_path)
    trainLabel1 = np.load(save_trainData_npy_path[:-4] + '_label.npy')
    save_trainData_npy_path = './trainData/AddWaterClassRiverSkinDetection2_22_38_57_68_77_86_90_100_105_112_115_123True_False.npy'
    trainData2 = np.load(save_trainData_npy_path)
    trainLabel2 = np.load(save_trainData_npy_path[:-4] + '_label.npy')
    save_trainData_npy_path = './trainData/AddWaterClassRiverSkinDetection3_22_38_57_68_77_86_90_100_105_112_115_123True_False.npy'
    trainData3 = np.load(save_trainData_npy_path)
    trainLabel3 = np.load(save_trainData_npy_path[:-4] + '_label.npy')
    trainData = np.concatenate([trainData1, trainData2, trainData3], axis=0)
    trainLabel = np.concatenate([trainLabel1, trainLabel2, trainLabel3], axis=0)
    print("begin!!!")
    print("trainData shape : ", trainData.shape)
    print("trainLabel shape : ", trainLabel.shape)
    trainDataset = MyDataset(trainData, trainLabel)
    # testDataset = MyDataset(testData, testLabel)
    # 训练一次 用到多少张图片（patch 像素块）32 64 128
    trainLoader = DataLoader(dataset=trainDataset, batch_size=mBatchSize, shuffle=True)
    # testLoader = DataLoader(dataset=testDataset, batch_size=mBatchSize, shuffle=True)
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
            img = img / torch.max(img, dim=1, keepdim=True)[0]
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
