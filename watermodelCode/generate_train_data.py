from materialNet import *
from utils.os_helper import mkdir
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

    trainData = np.array(trainData)
    trainLabel = np.array(trainLabel)
    testData = np.array(testData)
    testLabel = np.array(testLabel)
    np.save('./trainData.npy',trainData)
    np.save('./trainLabel.npy', trainLabel)
    np.save('./testData.npy', testData)
    np.save('./testLabel.npy', testLabel)

    # trainDataset = MyDataset(trainData, trainLabel)
    # testDataset = MyDataset(testData, testLabel)
    # trainLoader = DataLoader(dataset=trainDataset, batch_size=mBatchSize, shuffle=True)
    # testLoader = DataLoader(dataset=testDataset, batch_size=mBatchSize, shuffle=True)



