from materialNet import *
from model_block import network
import torch
import numpy as np
from utils.os_helper import mkdir
import time
import random
from utils.parse_args import parse_args
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
# CUDA:0

args = parse_args()
#lr, num_epochs, batch_size = 0.001, 200, 32
# lr, num_epochs, batch_size = 0.00050, 200, 16
mLearningRate = args.lr
mBatchSize = args.batchsize
mEpochs = args.epoch
model_path = args.model_path
print(mLearningRate)
print(mBatchSize)
print(mEpochs)
print(model_path)
# mBatchSize = 32 #尽量能够被训练总数整除
# mEpochs = 300
bands = 9
CLASSES_NUM=4
# #mLearningRate = 0.001
# mLearningRate = 0.0005
model_path = './SSRN_model_'+str(mBatchSize)+'_'+str(mLearningRate)+'/'
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
    trainData, trainLabel = generateData('train', 300, 11, DATA_TYPE)
    testData, testLabel = generateData('test', 300, 11, DATA_TYPE)
    # train_list, train_label = read_list(train_path)
    # test_list, test_label = read_list(test_path)
    # trainDataset = Dataset_patch_mem(train_list, train_label)
    # testDataset = Dataset_patch_mem(test_list, test_label)

    trainDataset = MyDataset(trainData, trainLabel)
    testDataset = MyDataset(testData, testLabel)
    trainLoader = DataLoader(dataset=trainDataset, batch_size=mBatchSize, shuffle=True)
    testLoader = DataLoader(dataset=testDataset, batch_size=mBatchSize, shuffle=True)

    # trainLoader = DataLoader(dataset=trainDataset, batch_size=mBatchSize, shuffle=True,num_workers=multiprocessing.cpu_count(), pin_memory = True)
    # testLoader = DataLoader(dataset=testDataset, batch_size=mBatchSize, shuffle=True,num_workers=multiprocessing.cpu_count(), pin_memory = True)
    # bt4 = time.time()
    # print('loda data time: ', bt4 - bt3)

    # model = ori_DBDA_network_MISH(bands,4,8).to(device)
    model = network.SSRN_network(bands, CLASSES_NUM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=mLearningRate, amsgrad=False)
    # model.load_state_dict(torch.load("/home/cjl/ywj_code/code/BS-NETs/bs_model/49.pkl"))
    # criterion=nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=mLearningRate)
    # 正则化
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
        # t1 = time.time()
        for i, data in enumerate(trainLoader, 0):
            t2 = time.time()
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
        tra_acc = trainCorrect.item() / trainTotal
        print('train epoch:', epoch, ' loss : ',float(trainLossTotal),' ; acc : ',tra_acc)
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
        model.eval()
        # 测试
        testCorrect = 0
        testTotal = 0
        for i, data in enumerate(testLoader, 0):
            img, label = data
            # img = torch.tensor(img)
            # label = torch.tensor(label)
            # img, label = torch.tensor(img).float().cuda(), torch.tensor(label).float().cuda()
            img, label = Variable(img).float().to(device), Variable(label).long().to(device)
            # tmp = torch.Tensor(img).to(mDevice)
            # img = img.permute(0, 3, 1, 2)
            img = img.permute(0, 2, 3, 1)
            img = img.unsqueeze(1)  # 增加通道维度
            testTotal += label.size(0)
            with torch.no_grad():
                predict = model(img)
                predict = torch.squeeze(predict)
                # 计算正确率
                predictIndex = torch.argmax(predict, dim=1)
                labelIndex = torch.argmax(label, dim=1)
                testCorrect += (predictIndex == labelIndex).sum()
        print('test epoch:', epoch, ' acc : ', testCorrect.item() / testTotal)
        print('\n')
        # if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), model_path + str(epoch) + '.pkl')
