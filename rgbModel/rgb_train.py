#输出每一类的召回率、准确率，还是只要准确率
# 损失函数、学习率啥的以哪个为准,纯DN值输入，
import sys
sys.path.append('../')
from utils.parse_args import parse_test_args
import torch.nn as nn
from zw_cnn import MaterialModel,MaterialModel_leakrelu
from fc_cnn import FcCNN
from mac_CNN import MAC_CNN
from model1 import Model1
import torch
import time
import numpy as np
import random
from utils.os_helper import mkdir
import pytorch_colors as colors
from sklearn.metrics import classification_report
from model_block.Dataset import MyDataset_ori
from torch.utils.data import DataLoader
from data.utilNetNew import *
args = parse_test_args()
# mLearningRate = args.lr
# mBatchSize = args.batchsize
# mEpochs = args.epoch
model_select =  args.model_select
mBatchSize=32
mEpochs=300
mLearningRate=0.0001
criterion=nn.MSELoss()
model_dict = {0:'zw_cnn',1:'fc_cnn',2:'mac_cnn',3:'zw_cnn_leakyrelu',4:'cnn-s'}
length_list = {0:11,1:32,2:16,3:11,4:32}
model_path = './'+model_dict[model_select]+'_model_'+str(mBatchSize)+'_'+str(mLearningRate)+'/'
# train_path = '/home/cjl/data/patch_sparse_cut_scale_9channel_data/_train_/'
# test_path = '/home/cjl/data/patch_sparse_cut_scale_9channel_data/_test_/'
device = torch.device('cuda')
mDevice=torch.device("cuda")
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
    mkdir(model_path)
    # bt2 = time.time()
    # print('pre process time: ', bt2 - bt)
    # train_list = os.listdir(train_path)
    # test_list = os.listdir(test_path)
    # n_sam = len(train_list)
    # n_sam2 =len(test_list)
    # bt3 = time.time()
    #先获取RGB-NIR
    #返回 B C H W
    trainData, trainLabel = generateRgbnirDNData('train', 300, length_list[model_select], DATA_TYPE)
    trainData = trainData.transpose(0, 2, 3, 1)  # BHW C
    #
    trainData_ = trainData.reshape(np.prod(trainData.shape[:3]), np.prod(trainData.shape[3:]))
    # trainData = trainData.reshape(trainData.shape[0]*trainData.shape[1],trainData.shape[2]*trainData.shape[3])
    scaler = preprocessing.StandardScaler()
    trainData_ = scaler.fit_transform(trainData_)
    trainData = trainData_.reshape(trainData.shape) # BHW C
    trainData = trainData.transpose(0, 3,1, 2) #B C H W
    #
    print(scaler.mean_, scaler.var_)


    testData, testLabel = generateRgbnirDNData('test', 300, length_list[model_select], DATA_TYPE)

    testData = testData.transpose(0, 2, 3, 1)  # BHW C
    testData_ = testData.reshape(np.prod(testData.shape[:3]), np.prod(testData.shape[3:]))
    testData_ = scaler.transform(testData_)
    testData = testData_.reshape(testData.shape)
    testData = testData.transpose(0,3,1,2)#B C H W
    if model_select==3:
        try:
            np.save('trainData.npy',trainData)
            np.save('trainLabel.npy', trainLabel)
            np.save('testData.npy',testData)
            np.save('testLabel.npy',testLabel)
        except:
            pass


    # train_list, train_label = read_list(train_path)
    # test_list, test_label = read_list(test_path)
    # trainDataset = Dataset_patch_mem(train_list, train_label)
    # testDataset = Dataset_patch_mem(test_list, test_label)

    trainDataset = MyDataset_ori(trainData, trainLabel)
    testDataset = MyDataset_ori(testData, testLabel)
    trainLoader = DataLoader(dataset=trainDataset, batch_size=mBatchSize, shuffle=True)
    testLoader = DataLoader(dataset=testDataset, batch_size=mBatchSize, shuffle=True)

    # trainLoader = DataLoader(dataset=trainDataset, batch_size=mBatchSize, shuffle=True,num_workers=multiprocessing.cpu_count(), pin_memory = True)
    # testLoader = DataLoader(dataset=testDataset, batch_size=mBatchSize, shuffle=True,num_workers=multiprocessing.cpu_count(), pin_memory = True)
    # bt4 = time.time()
    # print('loda data time: ', bt4 - bt3)

    # model = ori_DBDA_network_MISH(bands,4,8).to(device)
    if model_select==0:
        model = MaterialModel().to(device)
    elif model_select==1:
        model = FcCNN(4).to(device)
    elif model_select==2:
        model = MAC_CNN(4).to(device)
    elif model_select==3:
        model = MaterialModel_leakrelu().to(device)
    else:
        model = Model1(4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=mLearningRate)
    # model.load_state_dict(torch.load("/home/cjl/ywj_code/code/BS-NETs/bs_model/49.pkl"))
    criterion=nn.MSELoss()
    # criterion = nn.CrossEntropyLoss()
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

    # lr_adjust = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 15, eta_min=0.0, last_epoch=-1)
    # lr_adjust = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 15, eta_min=0.0, last_epoch=-1)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=8,
    #                                                        verbose=True, threshold=0.005, threshold_mode='rel',
    #                                                        cooldown=0,
    #                                                        min_lr=0, eps=1e-08)
    start_time = time.time()
    test_time = []
    for epoch in range(mEpochs):
        # 训练
        model.train()
        trainLossTotal = 0.0
        trainTotal = 0
        trainCorrect = 0

        train_label_list = []
        train_predict_list = []
        # t1 = time.time()
        for i, data in enumerate(trainLoader, 0):
            t2 = time.time()
            # print('pre data cost :',t2-t1)
            img, label = data
            # img = torch.tensor(img)
            # label = torch.tensor(label)
            # img = Variable(img).float().cuda()
            # img, label = Variable(img).float().to(device,non_blocking=True), Variable(label).long().to(device,non_blocking=True)
            img, label = Variable(img).float().to(device), Variable(label).float().to(device)#B C H W

            # img = img.permute(0, 2, 3, 1)
            # img = img.unsqueeze(1)  # 增加通道维度
            if model_select == 0 or model_select == 3:
                # img =
                tmp = torch.Tensor(img.size(0), 7, img.size(2), img.size(3)).to(device)
                # print(img[:, :3].size())
                # print(type(img[:, :3]))

                tmp[:, :3] = colors.rgb_to_yuv(img[:, :3])

                tmp[:, 4] = img[:, 0] - img[:, 1]
                tmp[:, 5] = img[:, 1] - img[:, 2]
                tmp[:, 6] = img[:, 2] - img[:, 3]
                tmp[:, 3] = img[:, 3]
            else:
                tmp = img[:,:3,:,:]

            trainTotal += label.size(0)
            predict = model(tmp)
            # t3 = time.time()
            # print('predict cost: ', t3 - t2)
            # 计算正确率 B C 1 1
            predict = predict.squeeze()
            predictIndex = torch.argmax(predict, dim=1)
            labelIndex = torch.argmax(label, dim=1)
            trainCorrect += (predictIndex == labelIndex).sum()
            # 产生loss
            loss = criterion(predict, label)  # + L2Loss(model, 1e-4)
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
            predictIndex = predictIndex.cpu().detach().numpy()
            labelIndex = labelIndex.cpu().detach().numpy()
            train_label_list.extend(labelIndex)
            train_predict_list.extend(predictIndex)

        tra_acc = trainCorrect.item() / trainTotal
        print('train epoch:', epoch, ' loss : ', float(trainLossTotal), ' ; acc : ', tra_acc)

        # train_label_list = np.array(train_label_list,dtype=np.int32).flatten()
        # train_predict_list = np.array(train_predict_list,dtype=np.int32).flatten()

        # 0:其他 1：皮肤，2：衣物，3：植物
        target_names = ['other', 'skin_', 'cloth', 'plant']
        res = classification_report(train_label_list, train_predict_list, target_names=target_names, output_dict=True)
        for k in target_names:
            # print(k,'skin pre=', res['skin']['precision'], 'skin rec=', res['skin']['recall'])
            print(k, ' pre=', res[k]['precision'], ' rec=', res[k]['recall'], ' f1-score=', res[k]['f1-score'])

        print('all train accuracy:', res['accuracy'], 'all train macro avg f1', res['macro avg']['f1-score'])
        print('\n')

        # scheduler.step(tra_acc)
        # scheduler.step()
        # lr_adjust.step()
        print('learning rate = ', optimizer.state_dict()['param_groups'][0]['lr'])
        # trainLossTotal = 0
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        model.eval()
        # 测试
        testCorrect = 0
        testTotal = 0
        test_label_list = []
        test_predict_list = []
        test_eopch_time = []
        for i, data in enumerate(testLoader, 0):
            img, label = data
            # img = torch.tensor(img)
            # label = torch.tensor(label)
            # img, label = torch.tensor(img).float().cuda(), torch.tensor(label).float().cuda()
            img, label = Variable(img).float().to(device), Variable(label).float().to(device)
            if model_select == 0 or model_select == 3:
                # img =
                tmp = torch.Tensor(img.size(0), 7, img.size(2), img.size(3)).to(device)
                # print(img[:, :3].size())
                # print(type(img[:, :3]))

                tmp[:, :3] = colors.rgb_to_yuv(img[:, :3])

                tmp[:, 4] = img[:, 0] - img[:, 1]
                tmp[:, 5] = img[:, 1] - img[:, 2]
                tmp[:, 6] = img[:, 2] - img[:, 3]
                tmp[:, 3] = img[:, 3]
            else:
                tmp = img[:,:3,:,:]
            # tmp = torch.Tensor(img).to(mDevice)
            # img = img.permute(0, 3, 1, 2)
            # img = img.permute(0, 2, 3, 1)
            # img = img.unsqueeze(1)  # 增加通道维度
            testTotal += label.size(0)

            with torch.no_grad():
                t_b = time.time()
                predict = model(tmp)
                predict = torch.squeeze(predict)
                # 计算正确率
                predictIndex = torch.argmax(predict, dim=1)
                t_e = time.time()
                t_pre = t_e-t_b
                test_eopch_time.append(t_pre)

                labelIndex = torch.argmax(label, dim=1)
                testCorrect += (predictIndex == labelIndex).sum()

                predictIndex = predictIndex.cpu().detach().numpy()
                labelIndex = labelIndex.cpu().detach().numpy()
                test_label_list.extend(labelIndex)
                test_predict_list.extend(predictIndex)
        test_eopch_time = np.array(test_eopch_time)
        test_time.append(np.nanmean(test_eopch_time))
        print('\n')
        print('test epoch:', epoch, ' acc : ', testCorrect.item() / testTotal)
        # test_label_list = np.array(test_label_list,dtype=np.int32).flatten()
        # test_predict_list = np.array(test_predict_list,dtype=np.int32).flatten()
        target_names = ['other', 'skin_', 'cloth', 'plant']
        res = classification_report(test_label_list, test_predict_list, target_names=target_names, output_dict=True)
        for k in target_names:
            # print(k,'skin pre=', res['skin']['precision'], 'skin rec=', res['skin']['recall'])
            print(k, ' pre=', res[k]['precision'], ' rec=', res[k]['recall'], ' f1-score=', res[k]['f1-score'])
        print('all test accuracy:', res['accuracy'], 'all test macro avg f1', res['macro avg']['f1-score'])
        print('\n')
        # if (epoch + 1) % 5 == 0:
        if model_select==0:
            torch.save(model.state_dict(), model_path + str(epoch) + '.pkl')
        else:
            if (epoch + 10) % 15 == 0:
                torch.save(model.state_dict(), model_path + str(epoch) + '.pkl')

    end_time = time.time()
    test_time = np.array(test_time)
    print("train cost time:",end_time - start_time)
    print('test a pix cost time:',np.nanmean(test_time))

