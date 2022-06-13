import torch
import torch.nn as nn
import numpy as np
import random
import torch.distributed as dist
from model_block.materialNet import MaterialSubModel
from model_block.PP_liteseg_final import PPLiteSeg, PPLiteAddSpectralSeg, PPLiteRgbCatSpectral
from utils.focalloss import focus_loss
from utils.cal_loss import focal_dice_loss
from utils.parse_args import parse_args
from model_block.Dataset import *
from data.dictNew import *
from torch.autograd import Variable
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'
# CUDA:0
from model_block.criterion import OhemCrossEntropy
# from materialNet import *
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import os
# import network
import time
from utils.init_seed import init_seeds
from utils.os_helper import mkdir
from torch.utils.data.distributed import DistributedSampler
from utils.mySampler import SequentialDistributedSampler
from sklearn.metrics import classification_report
from torch.cuda.amp import autocast, GradScaler
# 1) 初始化
dist.init_process_group(backend="nccl")

batch_size = 16
# mBatchSize = 16
bands = 9
CLASSES_NUM=4
# mLearningRate = 0.0005
model_save = './ddp_model/'
mkdir(model_save)


args = parse_args()
# batchsize 也可以改 目前为32
mBatchSize = args.batchsize

mEpochs = args.epoch
# model_select = args.model_select
mLearningRate = args.lr
mtrainBatchSize = args.batchsize #后期增加一个训练图变成38 整除2，或者其他
# mtestBatchSize = 2
# mEpochs = 300
# start = 5
# 分类准确率要用三个零
# mLearningRate = 0.001
# mLearningRate = 2.5e-05
mDevice=torch.device("cuda")

waterImgRootPath = '/home/cjl/ssd/dataset/shenzhen/img/train/'
waterLabelPath = '/home/cjl/ssd/dataset/shenzhen/label/Label_rename/'
png_path = '/home/cjl/ssd/dataset/shenzhen/rgb/needmark1/'
select_train_bands = [123,  98, 114, 100, 109, 112, 108, 102, 81, 125, 53, 92]  # 模型选择结果波段

# select_train_bands = [2,36,54,61,77,82,87,91,95,104,108]  # 手动选取波段
# select_train_bands = [x + 5 for x in select_train_bands]
input_bands_nums = len(select_train_bands)
cm_bin_sizes = [4, 8, 16]
spectral_inter_chs = [18, 24, 32, 64, 96]
nora = True
featureTrans = False
class_num = 2
# 这个参数最好大一些！！！
min_kept = args.min_kept
OhemCrossEntropy_thres = 0.9
band_sum = np.sum(np.array(select_train_bands))
model_path = './PPLiteRgbCatSpectral_AddHZ_' + str(min_kept) + '_' + str(mLearningRate) + '_' + \
             str(mtrainBatchSize) + '_' + str(band_sum) + '/'
# model_path = './PPLiteSeg_Spectral_hand_Select_Band' + str(min_kept) + '_' + str(mLearningRate) + '_' + \
#              str(mtrainBatchSize) + '_' + str(band_sum) + '/'
print(model_path)
print('mBatchSize',mtrainBatchSize)
print('mEpochs',mEpochs)
print('mLearningRate',mLearningRate)
# print('nora',nora)
print('featureTrans', featureTrans)
print('min_kept', min_kept)
print('nora',nora)

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
    init_seeds(local_rank+2021)
    # 问题完美解决！
    #保证每个进程有不同的随机种子，而又可以复现
    # init_seeds(1 + local_rank)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    trainFile = SeaFile + HangZhouTrain
    trainDataset = DatasetSpectralAndRgb(trainFile, waterImgRootPath, waterLabelPath, png_path, select_train_bands,
                                         nora, featureTrans)
    testDataset = DatasetSpectralAndRgb(HangZhouTest, waterImgRootPath, waterLabelPath, png_path, select_train_bands,
                                        nora, featureTrans)
    # trainLoader = DataLoader(dataset=trainDataset, batch_size=mtrainBatchSize, shuffle=True)
    # testLoader = DataLoader(dataset=testDataset, batch_size=mtrainBatchSize, shuffle=True)

    # dataset = RandomDataset(input_size, data_size)
    # 3）使用DistributedSampler
    trainLoader = DataLoader(dataset=trainDataset,
                             batch_size=batch_size,
                             sampler=DistributedSampler(trainDataset))
    test_sampler = SequentialDistributedSampler(testDataset,batch_size=batch_size)
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
    model = PPLiteRgbCatSpectral(num_classes=3, input_channel=3, spectral_input_channels=input_bands_nums,
                                 cm_bin_sizes=cm_bin_sizes, spectral_inter_chs=spectral_inter_chs)

    # 4) 封装之前要把模型移到对应的gpu
    # model.to(device)
    # 引入SyncBN，这句代码，会将普通BN替换成SyncBN。只用这一句代码就可以解决BN层问题
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    # criterion = nn.CrossEntropyLoss().to(device)
    criterion = OhemCrossEntropy(ignore_label=0,
                                 thres=OhemCrossEntropy_thres,
                                 min_kept=min_kept,
                                 weight=None,
                                 model_num_outputs=3,
                                 loss_balance_weights=[1, 1, 1])

    # optim.lr_scheduler.ReduceLROnPlateau
    scaler = GradScaler()


    ckpt_path = None
    # model.train()
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
                                                     output_device=local_rank,find_unused_parameters=True)
    # criterion = nn.SmoothL1Loss().to(device)
    # #reduce_tensor(torch.tensor(train_acc).cuda(args.local_rank)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


    # model.load_state_dict(torch.load("/home/cjl/ywj_code/code/BS-NETs/bs_model/49.pkl"))
    # criterion=nn.MSELoss()
    ## DDP: 要在构造DDP model之后，才能用model初始化optimizer。
    # optimizer = torch.optim.Adam(model.parameters(), lr=mLearningRate, amsgrad=False)
    # lr_adjust = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 15, eta_min=0.0, last_epoch=-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=mLearningRate)
    # optimizer = torch.optim.AdamW(model.parameters(),lr=mLearningRate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8, last_epoch=-1)
    lr_adjust = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=12,
                                                           verbose=True, threshold=0.005, threshold_mode='rel',
                                                           cooldown=0,
                                                           min_lr=0, eps=1e-08)
    t_start = time.time()
    mean = torch.tensor([0.5, 0.5, 0.5]).to(device)
    std = torch.tensor([0.5, 0.5, 0.5]).to(device)
    for epoch in range(300):
        model.train()
        trainLoader.sampler.set_epoch(epoch)
        trainLossTotal=0
        trainTotal = 0
        trainCorrect = 0
        for i, data in enumerate(trainLoader, 0):
            img, rgb_data, label = data  # label 需要改成 0，1，2，3，4 的形式 4表示其他
            img, rgb_data, label = img.float().to(device), rgb_data.float().to(device), label.long().to(device)
            img = img.permute(0, 3, 1, 2)
            rgb_data = rgb_data / 255.0
            rgb_data -= mean
            rgb_data /= std
            rgb_data = rgb_data.permute(0, 3, 1, 2)
            # with autocast():
            predict = model(rgb_data, img)  # B*CLASS_NUM*H*W
            losses = criterion(predict, label)
            torch.unsqueeze(losses, 0)
            # print(losses.shape)
            loss = losses.mean()
            # print(loss.shape)
            model.zero_grad()
            scaler.scale(loss).backward()
            trainLossTotal += loss.item()
            scaler.step(optimizer)
            scaler.update()
            # predict=model(img)
            predictIndex = torch.argmax(predict[0], dim=1)  # 计算一下准确率和召回率 B*H*W 和label1一样

            trainCorrect += torch.sum((predictIndex == label) & (label != 0)).item()
            trainTotal += torch.sum(label != 0).item()
            del predict
            del img
            del rgb_data, label
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
        train_acc = trainCorrect / trainTotal
        mean_loss = reduce_tensor(trainLossTotal).item()
        mean_acc = reduce_tensor(train_acc).item()
        if local_rank == 0:
            print('local_rank: ', local_rank, 'train epoch: ', epoch, 'loss :', mean_loss, ' acc:', mean_acc)
        # 为啥这步会卡死？？
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)
        # dist.barrier()
        lr_adjust.step(train_acc)
        if local_rank == 0:
            print('local_rank: ', local_rank,'learning rate = ', optimizer.state_dict()['param_groups'][0]['lr'])

        with torch.no_grad():
            model.eval()
            # 1. 得到本进程的prediction
            predictions = []
            labels = []
            test_total = 0
            test_correct = 0

            for i, data in enumerate(testLoader, 0):
                # img, rgb_data, label = data
                img, rgb_data, label = img.float().to(local_rank), rgb_data.float().to(local_rank), label.long().to(local_rank)
                img = img.permute(0, 3, 1, 2)
                rgb_data = rgb_data / 255.0
                rgb_data -= mean
                rgb_data /= std
                rgb_data = rgb_data.permute(0, 3, 1, 2)
                # data, label = data.float().to(local_rank), label.float().to(local_rank)
                # data = data.unsqueeze(1)
                predictions.append(model(rgb_data, img)[0])
                labels.append(label)
            # 进行gather 后面改成单点通信
            predictions = distributed_concat(torch.cat(predictions, dim=0),
                                             len(test_sampler.dataset))
            labels = distributed_concat(torch.cat(labels, dim=0),
                                        len(test_sampler.dataset))
            if local_rank == 0:
                predictions = predictions.squeeze()

                # predictIndex = torch.argmax(predictions, dim=1)
                # labelIndex = torch.argmax(labels, dim=1)

                predictIndex = torch.argmax(predictions, dim=1)  # 计算一下准确率和召回率 B*H*W 和label1一样
                test_correct += torch.sum((predictIndex == labels) & (labels != 0)).item()
                test_total += torch.sum(labels != 0).item()
                test_acc = test_correct/test_total

                print('local_rank: ', local_rank, 'test epoch: ', epoch,
                  ' acc:', test_acc.item())
                # predictIndex = predictIndex.cpu().detach().numpy().flatten()
                # labelIndex = labelIndex.cpu().detach().numpy().flatten()
                # target_names = ['other', 'skin', 'cloth', 'plant']
                # res = classification_report(labelIndex, predictIndex, target_names=target_names,
                #                             output_dict=True)
                # for k in target_names:
                #     # print(k,'skin pre=', res['skin']['precision'], 'skin rec=', res['skin']['recall'])
                #     print(k, ' pre=', res[k]['precision'], ' rec=', res[k]['recall'], ' f1-score=', res[k]['f1-score'])
                # print('all test accuracy:', res['accuracy'], 'all test macro avg f1', res['macro avg']['f1-score'])

            # 3. 现在我们已经拿到所有数据的predictioin结果，进行evaluate！
            # my_evaluate_func(predictions, labels)

        #通过torch.distributed.gather收集计算结果，进行test acc的计算,或者使用reduce_tensor计算两个GPU上的均值
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)
        # dist.barrier()
        if dist.get_rank() == 0:
            torch.save(model.module.state_dict(), model_save+"%d.ckpt" % epoch)
    t_end = time.time()
    print('train cost time: ',t_end-t_start)
            # print("Outside: input size", input_var.size(), "output_size", output.size())