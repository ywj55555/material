import sys
from materialNet import *
import network
import torch
import numpy as np
from utils.os_helper import mkdir
import time
import random
from utils.parse_args import parse_args
from torch.autograd import Variable
from DBDA_Conv import DBDA_network_MISH_full_conv
# from sklearn.metrics import classification_report
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
# CUDA:0

args = parse_args()
mLearningRate = args.lr
mBatchSize = args.batchsize
mEpochs = args.epoch
# model_save = './DBDA_model_'+str(mBatchSize)+'_'+str(mLearningRate)+'/'
device = torch.device('cuda')
mDevice=torch.device("cuda")
nora = args.nora
model_select = args.model_select
featureTrans = args.featureTrans#False#
# model_save = './IntervalSampleAddFeatureWaterModel_shenzhen2/'
# dataTypelist = ['water', 'water','water']
# dataType = dataTypelist[model_select-1]
dataType = 'sea'
print('featureTrans',featureTrans)

print('mBatchSize',mBatchSize)
print('mEpochs',mEpochs)
print('mLearningRate',mLearningRate)
print('model_select',model_select)
print('nora',nora)
# model_path = ['./IntervalSampleWaterModel/','./newSampleWaterModel/','./IntervalSampleBigWaterModel/']
model_kind = ["DBDA_full_conn", "DBDA_full_conv"]
model_save = "./" + model_kind[model_select - 1] + "_" + str(mBatchSize) + "_" + str(mLearningRate) + "_"+ str(featureTrans) + "/"
select_bands = [2,36,54,61,77,82,87,91,95,104,108]
select_bands = [x + 5 for x in  select_bands]

# hashBand = hash(select_bands)
# 换成小模型 11 通道 重新验证一下？？
# select_bands = [x for x in range(128)]
mkdir(model_save)
class_nums = 2

if featureTrans:
    bands_num = 21
else:
    bands_num = len(select_bands)

if bands_num != 128:
    bands_str = [str(x) for x in select_bands]
    bands_str = "_".join(bands_str)
else:
    bands_str = "128bands"


if __name__ == '__main__':
    bt = time.time()
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

    save_trainData_npy_path = './trainData/' + "big_32_0.001_False" + 'mulprocess.npy'
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
    if not os.path.exists(save_trainData_npy_path):
        # 数据的归一化 应该在分割完patch之后 避免以后需要不归一化的数据
        trainData, trainLabel = multiProcessGenerateData(dataType, 2000, 11, select_bands, nora=nora,
                                                         class_nums=class_nums,
                                                         intervalSelect=True, featureTrans=featureTrans)
        # trainData, trainLabel = generateData("dataType", 2500, 11, DATA_TYPE, nora=nora, class_nums=class_nums, intervalSelect = True, featureTrans = featureTrans)
        # trainData, trainLabel = generateData(dataType, 1000, 11, DATA_TYPE,nora=nora, class_nums = class_nums)

        # testData, testLabel = generateData(dataType, 600, 11, DATA_TYPE,nora=nora, class_nums=class_nums)
        # trainData = np.load('./trainData/train.npy')
        # trainLabel = np.load('./trainData/trainLabel.npy')
        # trainData = np.load('./trainData/trainIntervalAddFeature_1.npy')
        # trainLabel = np.load('./trainData/trainLabelIntervalAddFeature_1.npy')
        if not os.path.exists(save_trainData_npy_path):
            try:
                np.save(save_trainData_npy_path, trainData)
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

    # trainData, trainLabel = generateData('traintest', 300, 11, DATA_TYPE)
    # testData, testLabel = generateData('traintest', 300, 11, DATA_TYPE)
    # train_list, train_label = read_list(train_path)
    # test_list, test_label = read_list(test_path)
    # trainDataset = Dataset_patch_mem(train_list, train_label)
    # testDataset = Dataset_patch_mem(test_list, test_label)

    trainDataset = MyDataset(trainData, trainLabel)
    # testDataset = MyDataset(testData, testLabel)
    trainLoader = DataLoader(dataset=trainDataset, batch_size=mBatchSize, shuffle=True)
    # testLoader = DataLoader(dataset=testDataset, batch_size=mBatchSize, shuffle=True)

    # trainLoader = DataLoader(dataset=trainDataset, batch_size=mBatchSize, shuffle=True,num_workers=multiprocessing.cpu_count(), pin_memory = True)
    # testLoader = DataLoader(dataset=testDataset, batch_size=mBatchSize, shuffle=True,num_workers=multiprocessing.cpu_count(), pin_memory = True)
    # bt4 = time.time()
    # print('loda data time: ', bt4 - bt3)

    # model = ori_DBDA_network_MISH(bands,4,8).to(device)
    if model_select == 1:
        model = network.DBDA_network_MISH(bands_num, class_nums).to(device)
    else:
        model = DBDA_network_MISH_full_conv(bands_num, class_nums, 4).to(device)
    print("model name : ",model.name)
    optimizer = torch.optim.Adam(model.parameters(), lr=mLearningRate, amsgrad=False)
    # model.load_state_dict(torch.load("/home/cjl/ywj_code/code/BS-NETs/bs_model/49.pkl"))
    # criterion=nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=mLearningRate)
    # weight_decay_list = (param for name, param in model.named_parameters() if name[-4:] != 'bias' and "bn" not in name)
    # no_decay_list = (param for name, param in model.named_parameters() if name[-4:] == 'bias' or "bn" in name)
    # parameters = [{'params': weight_decay_list},
    #               {'params': no_decay_list, 'weight_decay': 0.}]
    #
    # optimizer = torch.optim.Adam(parameters, lr=mLearningRate, weight_decay=1e-4)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=mLearningRate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8, last_epoch=-1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15,T_mult=2)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 15, eta_min=0.0, last_epoch=-1)

    lr_adjust = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 15, eta_min=0.0, last_epoch=-1)
    # lr_adjust = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 15, eta_min=0.0, last_epoch=-1)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=8,
    #                                                        verbose=True, threshold=0.005, threshold_mode='rel',
    #                                                        cooldown=0,
    #                                                        min_lr=0, eps=1e-08)
    for epoch in range(mEpochs):
        # 训练
        model.train()
        trainLossTotal = 0.0
        trainTotal = 0
        trainCorrect = 0
        #
        # train_label_list = []
        # train_predict_list = []
        # t1 = time.time()
        for i, data in enumerate(trainLoader, 0):
            # t2 = time.time()
            # print('pre data cost :',t2-t1)
            img, label = data
            # img = torch.tensor(img)
            # label = torch.tensor(label)
            # img = Variable(img).float().cuda()
            # img, label = Variable(img).float().to(device,non_blocking=True), Variable(label).long().to(device,non_blocking=True)
            img, label = Variable(img).float().to(device), Variable(label).long().to(device)

            img = img.permute(0,2,3,1)
            img = img.unsqueeze(1)#增加通道维度

            trainTotal += label.size(0)
            predict = model(img)
            # t3 = time.time()
            # print('predict cost: ', t3 - t2)
            # 计算正确率 B C 1 1
            predict = predict.squeeze()
            predictIndex = torch.argmax(predict, dim=1)
            labelIndex = torch.argmax(label, dim=1)
            trainCorrect += (predictIndex == labelIndex).sum()
            # 产生loss
            loss = criterion(predict, labelIndex) #+ L2Loss(model, 1e-4)
            trainLossTotal += loss
            # print("loss = %.5f" % float(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # t1 = time.time()
            # print('loss back cost:', t1 - t3)
            # scheduler.step(epoch + i / len(trainLoader))

            # predict = model(img) #weight shape: B,self.bands,1,1
            #
            # # lossl2 = L2Loss(model,1e-4)
            # loss = criterion(predict, label) #交叉熵损失，还要加上权重正则化,在optim中包含了，或者在loss中包含也可以
            # # weight.squeeze()
            # trainLossTotal += loss.item()
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            # predictIndex = predictIndex.cpu().detach().numpy()
            # label = label.cpu().detach().numpy()
            # train_label_list.append(label)
            # train_predict_list.append(predictIndex)

        tra_acc = trainCorrect.item() / trainTotal
        print('train epoch:', epoch, ' loss : ', float(trainLossTotal), ' ; acc : ', tra_acc)
        #
        # train_label_list = np.array(train_label_list).flatten()
        # train_predict_list = np.array(train_predict_list).flatten()

        # 0:其他 1：皮肤，2：衣物，3：植物
        # target_names = ['other', 'skin', 'cloth', 'plant']
        # res = classification_report(train_label_list, train_predict_list, target_names=target_names, output_dict=True)
        # for k in target_names:
        #     # print(k,'skin pre=', res['skin']['precision'], 'skin rec=', res['skin']['recall'])
        #     print(k, ' pre=', res[k]['precision'], ' rec=', res[k]['recall'], ' f1-score=', res[k]['f1-score'])
        #
        # print('all train accuracy:',res['accuracy'],'all train macro avg f1', res['macro avg']['f1-score'])
        # print('\n')

        # scheduler.step(tra_acc)
        # scheduler.step()
        lr_adjust.step()
        print('learning rate = ', optimizer.state_dict()['param_groups'][0]['lr'])
        # trainLossTotal = 0
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        # model.eval()
        # 测试
        # testCorrect = 0
        # testTotal = 0
        # test_label_list = []
        # test_predict_list = []
        #
        # for i, data in enumerate(testLoader, 0):
        #     img, label = data
        #     # img = torch.tensor(img)
        #     # label = torch.tensor(label)
        #     # img, label = torch.tensor(img).float().cuda(), torch.tensor(label).float().cuda()
        #     img, label = Variable(img).float().to(device), Variable(label).long().to(device)
        #     # tmp = torch.Tensor(img).to(mDevice)
        #     # img = img.permute(0, 3, 1, 2)
        #     img = img.permute(0, 2, 3, 1)
        #     img = img.unsqueeze(1)  # 增加通道维度
        #     testTotal += label.size(0)
        #     with torch.no_grad():
        #         predict = model(img)
        #         predict = torch.squeeze(predict)
        #         # 计算正确率
        #         predictIndex = torch.argmax(predict, dim=1)
        #         labelIndex = torch.argmax(label, dim=1)
        #         testCorrect += (predictIndex == labelIndex).sum()
        #
        #         predictIndex = predictIndex.cpu().detach().numpy()
        #         label = label.cpu().detach().numpy()
        #         test_label_list.append(label)
        #         test_predict_list.append(predictIndex)
        #
        # print('\n')
        # print('test epoch:', epoch, ' acc : ', testCorrect.item() / testTotal)
        # test_label_list = np.array(test_label_list).flatten()
        # test_predict_list = np.array(test_predict_list).flatten()
        # target_names = ['other', 'skin', 'cloth', 'plant']
        # res = classification_report(test_label_list, test_predict_list, target_names=target_names, output_dict=True)
        # for k in target_names:
        #     # print(k,'skin pre=', res['skin']['precision'], 'skin rec=', res['skin']['recall'])
        #     print(k, ' pre=', res[k]['precision'], ' rec=', res[k]['recall'], ' f1-score=', res[k]['f1-score'])
        # print('all test accuracy:', res['accuracy'], 'all test macro avg f1', res['macro avg']['f1-score'])
        # print('\n')
        # if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), model_save + str(epoch) + '.pkl')
