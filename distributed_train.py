from materialNet import *
from utils.os_helper import mkdir
# import torch.distributed as dist
# import torch.multiprocessing as mp
import torch.distributed as dist
import argparse
from torch.utils.data.distributed import DistributedSampler
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
# CUDA:0

mBatchSize = 32
mEpochs = 300
#mLearningRate = 0.001
mLearningRate = 0.0001
mDevice=torch.device("cuda")
model_save = './ori_model_hz/'
mkdir(model_save)
if __name__ == '__main__':
    # 1) 初始化
    torch.distributed.init_process_group(backend="nccl")
    # 2） 配置每个进程的gpu
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    # world_size = 2
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--local_rank', type=int, default=-1)
    # # 添加必要参数
    # # local_rank：系统自动赋予的进程编号，可以利用该编号控制打印输出以及设置device
    #
    # torch.distributed.init_process_group(backend="nccl", init_method='file://shared/sharedfile',
    #                                      rank=parser.local_rank, world_size=world_size)

    # world_size：所创建的进程数，也就是所使用的GPU数量
    # （初始化设置详见参考文档）

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


    trainData, trainLabel = generateData('train_hz', 300, 11, DATA_TYPE)
    testData, testLabel = generateData('test_hz', 300, 11, DATA_TYPE)

    trainDataset = MyDataset(trainData, trainLabel)
    testDataset = MyDataset(testData, testLabel)
    trainLoader = DataLoader(dataset=trainDataset, batch_size=mBatchSize, shuffle=True)
    testLoader = DataLoader(dataset=testDataset, batch_size=mBatchSize, shuffle=True)

    model = MaterialSubModel(20, 4).cuda()

    # criterion=nn.MSELoss()
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=mLearningRate)

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
            predict = model(img)
            # 计算正确率
            predict = predict.squeeze()
            predictIndex = torch.argmax(predict, dim=1)
            labelIndex = torch.argmax(label, dim=1)
            trainCorrect += (predictIndex == labelIndex).sum()
            # 产生loss
            loss = criterion(predict, label)
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
        for i, data in enumerate(testLoader, 0):
            img, label = data
            # img, label = torch.tensor(img).float().cuda(), torch.tensor(label).float().cuda()
            img, label = Variable(img).float().cuda(), Variable(label).float().cuda()
            # tmp = torch.Tensor(img).to(mDevice)

            testTotal += label.size(0)
            predict = model(img)

            predict = torch.squeeze(predict)
            # 计算正确率
            predictIndex = torch.argmax(predict, dim=1)
            labelIndex = torch.argmax(label, dim=1)
            testCorrect += (predictIndex == labelIndex).sum()
        print('test epoch:', epoch, ': ', testCorrect.item() / testTotal)
        print('\n')
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), model_save+ str(epoch) + '.pkl')
