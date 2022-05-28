from materialNet import *
import os
from utils.os_helper import mkdir
from utils.parse_args import parse_args
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
# CUDA:0
args = parse_args()
mBatchSize = args.batchsize
mEpochs = args.epoch
#mLearningRate = 0.001
mLearningRate = args.lr
mDevice=torch.device("cuda")
model_save = './ori_model/'
nora = args.nora
print(mBatchSize)
print(mEpochs)
print(mLearningRate)
if nora:
    print("yes")
else:
    print("no")

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

    trainData, trainLabel = generateData('train', 300, 11, DATA_TYPE,nora=nora)
    testData, testLabel = generateData('test', 300, 11, DATA_TYPE,nora=nora)
    try:
        np.save('train.npy',trainData)
        np.save('trainLabel.npy', trainLabel)
        np.save('testData.npy',testData)
        np.save('testLabel.npy',testLabel)
    except:
        pass
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
        # if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), model_save+ str(epoch) + '.pkl')
