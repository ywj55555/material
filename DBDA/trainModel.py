from model_block.materialNet import *
import os
from utils.os_helper import mkdir
from utils.parse_args import parse_args
import numpy as np
import sys
from torch.autograd import Variable
from tqdm import tqdm
import time
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
# CUDA:0
args = parse_args()
# batchsize 也可以改 目前为32
mBatchSize = args.batchsize
train_set = args.train_set
mEpochs = args.epoch
model_select = args.model_select
mLearningRate = args.lr
dim = args.band_number
num_workers = args.num_workers
# 梯度更新步长 0.0001
mDevice=torch.device("cuda")
nora = args.nora
print('mBatchSize',mBatchSize)
print('mEpochs',mEpochs)
print('mLearningRate',mLearningRate)
print('model_select',model_select)
print('nora',nora)
model_size = ["small", "big"]
featureTrans = args.featureTrans

select_bands = [2,36,54,61,77,82,87,91,95,104,108]
select_bands = [x + 5 for x in  select_bands]
# dataType = 'sea'

class_nums = 2

if featureTrans:
    bands_num = 21
else:
    bands_num = len(select_bands)

bands_num = 128

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

model_save = "./" + "model_select_" + str(model_select) + "_" + str(mBatchSize) + "_" + str(mLearningRate) + "_" + str(intervalSelect) + "_" + str(nora) + \
             "_" + str(featureTrans) + "_" + activa + "/"

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

    dim = dim
    # train_set = '_all'#''#
    state = bands_str + '_' + str(intervalSelect) + train_set#
    model_save = model_save + state + '/'
    model_save = model_save + 'dim_' + str(dim) + '/'

    save_trainData_npy_path = './trainData/' + state + '.npy'
    print(save_trainData_npy_path)
    if not os.path.exists(save_trainData_npy_path):
        # 数据的归一化 应该在分割完patch之后 避免以后需要不归一化的数据
        # trainData, trainLabel = multiProcessGenerateData('train', 2500, 11, activate, nora=nora, class_nums=class_nums,
        #                                                  intervalSelect=intervalSelect, featureTrans=featureTrans)
        trainData, trainLabel = generateData(train_set,'train', 2500, 11, nora=nora, class_nums=class_nums,
                                                         intervalSelect=intervalSelect, featureTrans=False)
        if not os.path.exists(save_trainData_npy_path):
            try:
                np.save(save_trainData_npy_path,trainData)
                np.save(save_trainData_npy_path[:-4] + '_label.npy', trainLabel)
                # np.save('./testData/testData.npy',testData)
                # np.save('./testData/testLabel.npy',testLabel)
            except:
                print("error")
                sys.exit()
    else:
        trainData = np.load(save_trainData_npy_path)
        trainLabel = np.load(save_trainData_npy_path[:-4] + '_label.npy')
        print("train data exist!!!")
    print("begin!!!")
    print("trainData shape : ", trainData.shape)
    print("trainLabel shape : ", trainLabel.shape)

    save_testData_npy_path = './testData/' + state + '.npy'
    print(save_testData_npy_path)
    if not os.path.exists(save_testData_npy_path):
        # 数据的归一化 应该在分割完patch之后 避免以后需要不归一化的数据
        # testData, testLabel = multiProcessGenerateData('test', 500, 11, activate, nora=nora, class_nums=class_nums,
        #                                                  intervalSelect=intervalSelect, featureTrans=featureTrans)
        testData, testLabel = generateData(train_set,'test', 500, 11, nora=nora, class_nums=class_nums,
                                                         intervalSelect=intervalSelect, featureTrans=False)
        if not os.path.exists(save_testData_npy_path):
            try:
                np.save(save_testData_npy_path,testData)
                np.save(save_testData_npy_path[:-4] + '_label.npy', testLabel)
                # np.save('./testData/testData.npy',testData)
                # np.save('./testData/testLabel.npy',testLabel)
            except:
                print("error")
                sys.exit()
    else:
        testData = np.load(save_testData_npy_path)
        testLabel = np.load(save_testData_npy_path[:-4] + '_label.npy')
        print("test data exist!!!")
    print("begin!!!")
    print("testData shape : ", testData.shape)
    print("testLabel shape : ", testLabel.shape)
#############
    # trainData = np.load('train_128_2.npy')#_21
    # trainLabel = np.load('trainLabel_128_2.npy')
    # testData = np.load('testData_128_2.npy')
    # testLabel = np.load('testLabel_128_2.npy')
##############
#     trainData, trainLabel = generateData('train', 3500, 11, DATA_TYPE,nora=nora)
#     testData, testLabel = generateData('test', 3500, 11, DATA_TYPE,nora=nora)
#     try:
#         np.save('train_128_2.npy',trainData)
#         np.save('trainLabel_128_2.npy', trainLabel)
#         np.save('testData_128_2.npy',testData)
#         np.save('testLabel_128_2.npy',testLabel)
#     except:
#         pass
# ##############
    trainDataset = MyDataset(train_set, trainData, trainLabel, dim = dim, feature_extraction=featureTrans, dataType = 'train', size=11)
    testDataset = MyDataset(train_set, testData, testLabel, dim = dim, feature_extraction=featureTrans, dataType = 'test', size=11)

    trainLoader = DataLoader(dataset=trainDataset, batch_size=1, shuffle=True, num_workers = num_workers)
    testLoader = DataLoader(dataset=testDataset, batch_size=1, shuffle=True, num_workers = 10)
    if featureTrans == True:
        dim = 21
    # 
    if model_select==1:
        model = MaterialSubModel(dim, 2).cuda()
    elif model_select==2:
        model = CNN(dim, 2).cuda()
    elif model_select==3:
        from DBDA_net import DBDA_network_with_2D
        model = DBDA_network_with_2D(dim, 2).cuda()
    elif model_select==4:
        from DBDA_net import DBDA_network_without_attention
        model = DBDA_network_without_attention(dim, 2).cuda()
    elif model_select==5:
        from DBDA_net import DBDA_network_three_losses
        model = DBDA_network_three_losses(dim, 2).cuda()
        gama = 0.1
    elif model_select==6:
        from DBDA_net import DBDA_network_three_losses_cross_Attention
        model = DBDA_network_three_losses_cross_Attention(dim, 2).cuda()
        gama = 0.1
    # print('Begin!!')
    # criterion=nn.MSELoss()
    # criterion = nn.SmoothL1Loss()
    criterion =torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=mLearningRate)
    optimizer = torch.optim.SGD(model.parameters(), lr=mLearningRate, momentum = 0.1, weight_decay = 0.0001)

    for epoch in range(mEpochs):
        # 训练
        # model.train()
        trainCorrect = 0
        trainCorrect_spec = 0
        trainCorrect_spat = 0
        trainTotal = 0
        trainLossTotal = 0.0
        for i, data in enumerate(tqdm(trainLoader)):

            img, label = data
            img, label = Variable(img).float().cuda().squeeze(), Variable(label).float().cuda().squeeze()
            # img, label = torch.tensor(img).float().cuda().squeeze(), torch.tensor(label).float().cuda().squeeze()
            trainTotal += label.size(0)
            # print(img.shape)
            step = (label.size(0) // mBatchSize) + 1
            for i in range(step - 1):
                if i == step - 2:
                    img_batch = img[int(i*mBatchSize):,:,:,:]
                    label_batch = label[int(i*mBatchSize):,:]

                else:
                    img_batch = img[int(i*mBatchSize):int((i+1)*mBatchSize),:,:,:]
                    label_batch = label[int(i*mBatchSize):int((i+1)*mBatchSize),:]
                # print(img_batch.shape)
                labelIndex = torch.argmax(label_batch, dim=1)
                if model_select == 5 or model_select == 6:
                    predict, predict_spec, predict_spat = model(img_batch)
                    predictIndex_spec = torch.argmax(predict_spec, dim=1)
                    trainCorrect_spec += (predictIndex_spec == labelIndex).sum()
                    loss_spec = criterion(predict_spec, labelIndex) * gama
                    predictIndex_spat = torch.argmax(predict_spat, dim=1)
                    trainCorrect_spat += (predictIndex_spat == labelIndex).sum()
                    loss_spat = criterion(predict_spat, labelIndex) * gama
                else:
                    predict = model(img_batch)

                # print(label_batch)
            # 计算正确率
                predict = predict.squeeze()
                predictIndex = torch.argmax(predict, dim=1)
                trainCorrect += (predictIndex == labelIndex).sum()
                # 产生loss
                loss = criterion(predict, labelIndex)
                if model_select == 5 or model_select == 6:
                    loss = loss + loss_spec + loss_spat
                # print(loss)
                # print(predict.shape)
                # print(labelIndex.shape)

                trainLossTotal += loss
            # print("loss = %.5f" % float(loss))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        print('train epoch:', epoch, ': ', trainCorrect.item() / trainTotal)
        print('total loss = %.5f' % float(trainLossTotal))
        if model_select == 5 or model_select == 6:
            print('train epoch:', epoch, 'spec_acc: ', trainCorrect_spec.item() / trainTotal)
            print('train epoch:', epoch, 'spat_acc: ', trainCorrect_spat.item() / trainTotal)
        # torch.save(model.state_dict(), "model/concat11/params" + str(epoch) + ".pkl")


        # 测试
        if (epoch+1) % 3 == 0:
            testCorrect = 0
            testCorrect_spec = 0
            testCorrect_spat = 0
            testTotal = 0
            # model.eval()
            for i, data in enumerate(tqdm(testLoader)):
                img, label = data
                # img, label = torch.tensor(img).float().cuda(), torch.tensor(label).float().cuda()
                img, label = Variable(img).float().cuda().squeeze(), Variable(label).float().cuda().squeeze()
                # tmp = torch.Tensor(img).to(mDevice)

                testTotal += label.size(0)
                # print(img.shape)
                labelIndex = torch.argmax(label, dim=1)
                if model_select == 5 or model_select == 6:
                    predict, predict_spec, predict_spat = model(img)
                    predict_spec = torch.squeeze(predict_spec)
                    predict_spat = torch.squeeze(predict_spat)
                    predictIndex_spec = torch.argmax(predict_spec, dim=1)
                    testCorrect_spec += (predictIndex_spec == labelIndex).sum()
                    predictIndex_spat = torch.argmax(predict_spat, dim=1)
                    testCorrect_spat += (predictIndex_spat == labelIndex).sum()
                else:
                    predict = model(img)
                predict = torch.squeeze(predict)
                # 计算正确率
                predictIndex = torch.argmax(predict, dim=1)
                testCorrect += (predictIndex == labelIndex).sum()
            print('test epoch:', epoch, ': ', testCorrect.item() / testTotal)
            if model_select == 5 or model_select == 6:
                print('test epoch:', epoch, 'spec_acc: ', testCorrect_spec.item() / testTotal)
                print('test epoch:', epoch, 'spat_acc: ', testCorrect_spat.item() / testTotal)
            print('\n')
            # if (epoch + 1) % 5 == 0:
        if (epoch+1) % 3 == 0:
            mkdir(model_save)
            torch.save(model.state_dict(), model_save+ str(epoch) + 'test_acc_' + str(testCorrect.item() / testTotal) + '.pkl', _use_new_zipfile_serialization=False)
