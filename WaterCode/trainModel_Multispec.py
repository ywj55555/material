from Dataloader_Multispec import *
import os
from utils.os_helper import mkdir
from utils.parse_args import parse_args
import numpy as np
import sys
from torch.autograd import Variable
from tqdm import tqdm
from model_block.criterion import OhemCrossEntropy,CrossEntropy
from torch.cuda.amp import autocast, GradScaler
from model_block.PP_liteseg_final import PPLiteSeg
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
select_train_bands = [114, 109, 125,  53, 108,  81, 100, 112,  25,  90,  96, 123 ]
# dataType = 'sea'

#class_nums = 31

min_kept = args.min_kept
OhemCrossEntropy_thres = 0.9

mean = torch.tensor([0.5, 0.5, 0.5]).cuda()
std = torch.tensor([0.5, 0.5, 0.5]).cuda()

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


activa = args.activa
print('activa :', activa)
from utils.load_spectral import Sigmoid
from utils.load_spectral import Tanh
if activa == 'sig':
    activate = Sigmoid
else:
    activate = Tanh

model_save = "./" + "Multispec_" + "model_select_" + str(model_select) + "_" + str(mBatchSize) + "_" + str(mLearningRate) + "/"

def adjust_learning_rate(optimizer, base_lr, max_iters,
        cur_iters, power=0.9, nbb_mult=10):
    lr = base_lr*((1-float(cur_iters)/max_iters)**(power))
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) == 2:
        optimizer.param_groups[1]['lr'] = lr * nbb_mult
    return lr

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
    model_save = model_save + 'dim_' + str(dim) + train_set + '/'

    trainDataset = MyDataset_whole(train_set, dim = dim, feature_extraction=featureTrans, dataType = 'train')
    testDataset = MyDataset_whole(train_set, dim = dim, feature_extraction=featureTrans, dataType = 'test')

    trainLoader = DataLoader(dataset=trainDataset, batch_size=mBatchSize, shuffle=True, num_workers = 10)#, pin_memory=False
    testLoader = DataLoader(dataset=testDataset, batch_size=mBatchSize, shuffle=True, num_workers = 10)#, pin_memory=False
    # prefetcher = DataPrefetcher(trainLoader)
    dim = 128
    #
    class_nums = 4
    if model_select==1:
        model = MaterialSubModel(dim, class_nums+1).cuda()
    elif model_select==2:
        model = CNN(dim, 2).cuda()
    elif model_select==3:
        from Two_CNN import Two_CNN
        model = Two_CNN(dim, class_nums+1).cuda()
    elif model_select==4:
        from DBDA_net import DBDA_network_without_attention
        model = DBDA_network_without_attention(dim, class_nums+1).cuda()
    elif model_select==5:
        from DBDA_net import DBDA_network_three_losses
        model = DBDA_network_three_losses(dim, class_nums+1).cuda()
        gama = 0.1
    elif model_select==6:
        from DBDA_net import DBDA_network_three_losses_cross_Attention
        model = DBDA_network_three_losses_cross_Attention(dim, class_nums+1).cuda()
        gama = 0.1
    elif model_select==7:
        model = PPLiteSeg(class_nums+1, dim).cuda()
    elif model_select == 8:
        from DBDA_net import DBDA_with_only_spatial
        model = DBDA_with_only_spatial(dim, class_nums+1).cuda()
    # print('Begin!!')
    # criterion=nn.MSELoss()
    # criterion = nn.SmoothL1Loss()
    # criterion =torch.nn.CrossEntropyLoss()
    # criterion = OhemCrossEntropy(ignore_label=0,
    #                              thres=OhemCrossEntropy_thres,
    #                              min_kept=min_kept,
    #                              weight=None,
    #                              model_num_outputs=1,
    #                              loss_balance_weights=[1])
    criterion = CrossEntropy(ignore_label=0,
                                 weight=None,
                                 model_num_outputs=1,
                                 loss_balance_weights=[1])
    if model_select==7:
        criterion = OhemCrossEntropy(ignore_label=0,
                                     thres=OhemCrossEntropy_thres,
                                     min_kept=min_kept,
                                     weight=None,
                                     model_num_outputs=3,
                                     loss_balance_weights=[1,1,1])
    optimizer = torch.optim.Adam(model.parameters(), lr=mLearningRate, weight_decay = 0.0001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=mLearningRate, momentum = 0.1, weight_decay = 0.0001)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=12,
    #                                           verbose=True, threshold=0.005, threshold_mode='rel', cooldown=0,
    #                                           min_lr=0, eps=1e-08)
    scaler = GradScaler()
    model.train()
    for epoch in range(mEpochs):
        # 训练
        # model.train()
        trainCorrect = 0
        trainCorrect_spec = 0
        trainCorrect_spat = 0
        trainTotal = 0
        trainLossTotal = 0.0
        # import time
        # time1 = time.time()
        # img, label = prefetcher.next()
        # iteration = 0
        # while img is not None:
        #     iteration += 1
        #     if iteration % 50 == 0:
        #         time2 = time.time()
        #         print(str(iteration) + ':' +str((time2-time1)/60))
            # 训练代码
#        for i, data in enumerate(tqdm(trainLoader)):
        for i, data in enumerate(tqdm(trainLoader)):
            img_A, label_A = data
            # img_A, label_A = Variable(img).float().cuda(), Variable(label).long().cuda()
            sep = 5
            _,_,H,W = img_A.shape
            H_Sep = H // sep
            W_Sep = W // sep
            for i in range(sep):
                for j in range(sep):
                    img = img_A[:,:,i*H_Sep:(i+1)*H_Sep,j*W_Sep:(j+1)*W_Sep]
                    label = label_A[:,i*H_Sep:(i+1)*H_Sep,j*W_Sep:(j+1)*W_Sep]
                    img, label = Variable(img).float().cuda(), Variable(label).long().cuda()
                    if model_select == 5 or model_select == 6:
                        # with autocast():
                        predict, predict_spec, predict_spat = model(img)
                        predictIndex_spec = torch.argmax(predict_spec, dim=1)
                        trainCorrect_spec += ((predictIndex_spec == label) & (label != 0)).sum()
                        loss_spec = criterion(predict_spec, label) * gama
                        predictIndex_spat = torch.argmax(predict_spat, dim=1)
                        trainCorrect_spat += ((predictIndex_spat == label) & (label != 0)).sum()
                        loss_spat = criterion(predict_spat, label) * gama
                    elif model_select == 7:
                        # with autocast():
                        predict = model(img)
                        loss = criterion(predict, label)
                        predictIndex = torch.argmax(predict[0], dim=1)
                    else:
                        predict = model(img)

                        # 产生loss
                    if model_select!= 7:
                        predictIndex = torch.argmax(predict, dim=1)
                        loss = criterion(predict, label)#_sep_sep

                    if model_select == 5 or model_select == 6:
                        loss = loss + loss_spec + loss_spat

                    trainLossTotal += loss
                    trainCorrect += ((predictIndex == label) & (label != 0)).sum()
                    trainTotal += torch.sum(label != 0).item()
                    print(loss)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    # optimizer.zero_grad()
                    # loss.backward()
                    # optimizer.step()
            # img, label = prefetcher.next()
        accuracy = trainCorrect.item() / trainTotal
        print('train epoch:', epoch, ': ', accuracy)
        # scheduler.step(accuracy)
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
                img, label = Variable(img).float().cuda(), Variable(label).long().cuda()
                # tmp = torch.Tensor(img).to(mDevice)
                # if 1:
                #     img = img.sum(1)
#                with autocast():
                if model_select == 5 or model_select == 6:
                    predict, predict_spec, predict_spat = model(img)
                    predictIndex_spec = torch.argmax(predict_spec, dim=1)
                    testCorrect_spec += ((predictIndex_spec == label) & (label != 0)).sum()##
                    predictIndex_spat = torch.argmax(predict_spat, dim=1)
                    testCorrect_spat += ((predictIndex_spat == label) & (label != 0)).sum()
                else:
                    predict = model(img)
                # 计算正确率
                predictIndex = torch.argmax(predict, dim=1)
                testCorrect += ((predictIndex == label) & (label != 0)).sum()
                testTotal += torch.sum(label != 0).item()
            print('test epoch:', epoch, ': ', testCorrect.item() / testTotal)
            if model_select == 5 or model_select == 6:
                print('test epoch:', epoch, 'spec_acc: ', testCorrect_spec.item() / testTotal)#
                print('test epoch:', epoch, 'spat_acc: ', testCorrect_spat.item() / testTotal)#
            print('\n')
            # if (epoch + 1) % 5 == 0:
        if (epoch+1) % 3 == 0:
            if model_select == 5 or model_select == 6:
                mkdir(model_save)
                torch.save(model.state_dict(),model_save + str(epoch) + '_test_acc_' + str(testCorrect.item() / testTotal) + '_spec_acc_' + str(testCorrect_spec.item() / testTotal) + '_spat_acc_' + str((testCorrect_spat.item() / testTotal)) + '.pkl', _use_new_zipfile_serialization=False)
            else:
                mkdir(model_save)
                torch.save(model.state_dict(), model_save+ str(epoch) + 'test_acc_' + str(testCorrect.item() / testTotal) + '.pkl', _use_new_zipfile_serialization=False)

