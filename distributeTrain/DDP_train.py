import torch.distributed as dist
from model_block.materialNet import *
from torch.utils.data import DataLoader
import time
from utils.init_seed import init_seeds
from utils.os_helper import mkdir
from torch.utils.data.distributed import DistributedSampler
from utils.mySampler import SequentialDistributedSampler
# 1) 初始化
dist.init_process_group(backend="nccl")

batch_size = 16
mBatchSize = 16
model_save = './ddp_model/'
mkdir(model_save)
# data_size = 90

#用于平均loss和acc
def reduce_tensor(tensor: torch.Tensor) :
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= torch.distributed.get_world_size()#总进程数
    return rt
# 合并结果的函数
# 1. all_gather，将各个进程中的同一份数据合并到一起。
#   和all_reduce不同的是，all_reduce是平均，而这里是合并。
# 2. 要注意的是，函数的最后会裁剪掉后面额外长度的部分，这是之前的SequentialDistributedSampler添加的。
# 3. 这个函数要求，输入tensor在各个进程中的大小是一模一样的。
#最好换成点对点通信
def distributed_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat[:num_total_examples]

#ProcessGroupNCCL does not support gather 通过send/recv实现
def distributed_concat_master(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.gather(tensor,output_tensors)
    #通过send和recv代替
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat[:num_total_examples]

if __name__ == '__main__':

    # seed = 2021
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # random.seed(seed)
    # # 可以考虑改变benchmark为true
    # torch.backends.cudnn.benchmark = False
    # # 配合随机数种子，确保网络多次训练参数一致
    # torch.backends.cudnn.deterministic = True
    # # 使用非确定性算法
    # torch.backends.cudnn.enabled = True
    # 2） 配置每个进程的gpu
    local_rank = torch.distributed.get_rank()
    init_seeds(local_rank + 2021)
    # init_seeds(local_rank+1)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # trainData, trainLabel = generateData('traintt', 300, 11, DATA_TYPE)
    trainData = np.load('trainData.npy')
    trainLabel = np.load('trainLabel.npy')
    testData = np.load('testData.npy')
    testLabel = np.load('testLabel.npy')

    # trainData = np.load()
    # testData, testLabel = generateData('test', 300, 11, DATA_TYPE)

    trainDataset = MyDataset(trainData, trainLabel)
    testDataset = MyDataset(testData, testLabel)
    # trainLoader = DataLoader(dataset=trainDataset, batch_size=mBatchSize, shuffle=True)
    # testLoader = DataLoader(dataset=testDataset, batch_size=mBatchSize, shuffle=True)

    # dataset = RandomDataset(input_size, data_size)
    # 3）使用DistributedSampler
    trainLoader = DataLoader(dataset=trainDataset,
                             batch_size=batch_size,
                             sampler=DistributedSampler(trainDataset))
    test_sampler = SequentialDistributedSampler(testDataset,batch_size=16)
    testLoader =DataLoader(dataset=testDataset,
                             batch_size=batch_size,
                             sampler=test_sampler)

    # class Model(nn.Module):
    #     def __init__(self, input_size, output_size):
    #         super(Model, self).__init__()
    #         self.fc = nn.Linear(input_size, output_size)
    #
    #     def forward(self, input):
    #         output = self.fc(input)
    #         print("  In Model: input size", input.size(),
    #               "output size", output.size())
    #         return output

    model = MaterialSubModel(20, 4)

    # 4) 封装之前要把模型移到对应的gpu
    model.to(device)
    # 引入SyncBN，这句代码，会将普通BN替换成SyncBN。只用这一句代码就可以解决BN层问题
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    ckpt_path = None
    # DDP: Load模型要在构造DDP模型之前，且只需要在master上加载就行了。
    if dist.get_rank() == 0 and ckpt_path is not None:
        dist.barrier()
        model.load_state_dict(torch.load(ckpt_path))
        dist.barrier()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # 5) 封装
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[local_rank],
                                                     output_device=local_rank)
    criterion = nn.SmoothL1Loss().to(device)
    #reduce_tensor(torch.tensor(train_acc).cuda(args.local_rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    t_start= time.time()
    for epoch in range(200):
        trainLoader.sampler.set_epoch(epoch)
        trainLossTotal=0
        trainTotal = 0
        trainCorrect = 0
        for data,label in trainLoader:
            if torch.cuda.is_available():
                imgdata = data.float().to(device)
                label = label.float().to(device)
            else:
                imgdata = data
                label = label.float()
            # input_var
            predict = model(imgdata)
            # print('local_rank: ',local_rank,' batch size:',predict.size()[0])
            # predict = predict.squeeze()
            trainTotal += label.size(0)
            predict = predict.squeeze()
            predictIndex = torch.argmax(predict, dim=1)
            labelIndex = torch.argmax(label, dim=1)
            trainCorrect += (predictIndex == labelIndex).sum()
            loss = criterion(predict, label)
            #reduce_tensor(torch.tensor(train_acc).cuda(args.local_rank)
            trainLossTotal += loss
            # print("loss = %.5f" % float(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_acc = trainCorrect / trainTotal
        # print('local_rank: ',local_rank,'train epoch: ',epoch,'loss :',trainLossTotal.item(),' acc:',train_acc)


        mean_loss =  reduce_tensor(trainLossTotal).item()
        mean_acc = reduce_tensor(train_acc).item()
        if local_rank==0:
            print('local_rank: ', local_rank, 'train epoch: ', epoch, 'mean_loss :', mean_loss, ' mean_acc:', mean_acc)
        #为啥这步会卡死？？
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)
        with torch.no_grad():
            # 1. 得到本进程的prediction
            predictions = []
            labels = []
            test_total = 0
            test_correct = 0
            # print('local_rank', local_rank, 'begin test!!!! ')
            for data, label in testLoader:
                data, label = data.float().to(local_rank), label.float().to(local_rank)
                predictions.append(model(data))
                labels.append(label)
            # 进行gather 后面改成单点通信,nccl不支持
            # if local_rank==0:
            # print('local_rank', local_rank, 'ending test!!!! ')
            # print('local_rank', local_rank, 'before length of pre: ', len(predictions)*16)
            predictions = distributed_concat(torch.cat(predictions, dim=0),
                                                 len(test_sampler.dataset))
            labels = distributed_concat(torch.cat(labels, dim=0),
                                            len(test_sampler.dataset))
                # predictions2 =
            # predictions = torch.cat(predictions,dim=0)
            # predictions2 = torch.zeros_like(predictions)
            # req = torch.distributed.irecv(predictions2, 1)
            # req.wait()
            # print('local_rank',local_rank,'after length of pre: ',predictions.size(0))
            # predictions = torch.cat([predictions,predictions2],dim=0)

            predictions = predictions.squeeze()

            predictIndex = torch.argmax(predictions, dim=1)
            labelIndex = torch.argmax(labels, dim=1)

            test_total += labelIndex.size(0)
            test_correct += (predictIndex == labelIndex).sum()
            test_acc = test_correct/test_total
            if local_rank==0:
                print('local_rank: ', local_rank, 'test epoch: ', epoch,
                  ' acc:', test_acc.item())
            # else:
            #     predictions = torch.cat(predictions,dim=0)
            #     req = torch.distributed.isend(predictions, 0)
            #     req.wait()
            #     print('local_rank', local_rank, 'length of pre: ', predictions.size(0))
            # 3. 现在我们已经拿到所有数据的predictioin结果，进行evaluate！
            # my_evaluate_func(predictions, labels)

        #通过torch.distributed.gather收集计算结果，进行test acc的计算,或者使用reduce_tensor计算两个GPU上的均值
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)
        if dist.get_rank() == 0:
            torch.save(model.module.state_dict(), model_save+"%d.ckpt" % epoch)
    t_end = time.time()
    print('train cost time: ',t_end-t_start)
            # print("Outside: input size", input_var.size(), "output_size", output.size())