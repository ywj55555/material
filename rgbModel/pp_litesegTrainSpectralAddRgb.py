import sys
sys.path.append('../')
from model_block.materialNet import MaterialSubModel
from model_block.PP_liteseg_final import PPLiteSeg, PPLiteAddSpectralSeg
from utils.os_helper import mkdir
from utils.focalloss import focus_loss
from utils.cal_loss import focal_dice_loss
from utils.parse_args import parse_args
from model_block.Dataset import *
from data.dictNew import *
from torch.autograd import Variable
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'
# CUDA:0
from model_block.criterion import OhemCrossEntropy
from torch.cuda.amp import autocast, GradScaler


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
spectral_inter_chs = [24, 32, 64, 96, 128]
nora = True
featureTrans = False
class_num = 2
# 这个参数最好大一些！！！
min_kept = args.min_kept
OhemCrossEntropy_thres = 0.9
band_sum = np.sum(np.array(select_train_bands))
model_path = './PPLiteSeg_SpectralAddRGB' + str(min_kept) + '_' + str(mLearningRate) + '_' + \
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
mean = torch.tensor([0.5, 0.5, 0.5]).cuda()
std = torch.tensor([0.5, 0.5, 0.5]).cuda()

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
    mkdir(model_path)

    trainDataset = DatasetSpectralAndRgb(SeaFile, waterImgRootPath, waterLabelPath, png_path, select_train_bands, nora, featureTrans)
    # testDataset = Dataset_all(testFile_hz,train_data)
    trainLoader = DataLoader(dataset=trainDataset, batch_size=mtrainBatchSize, shuffle=True)
    # testLoader = DataLoader(dataset=testDataset, batch_size=mtestBatchSize, shuffle=True)

    # model = PPLiteSeg(num_classes=3, input_channel=3, cm_bin_sizes=cm_bin_sizes).cuda() # PPLiteAddSpectralSeg
    model = PPLiteAddSpectralSeg(num_classes=3, input_channel=3, spectral_input_channels=input_bands_nums,
                                 cm_bin_sizes=cm_bin_sizes, spectral_inter_chs=spectral_inter_chs).cuda()
    # model = PPLiteSeg(num_classes=3, input_channel=input_bands_nums).cuda()
    # model = MaterialSubModel(in_channels=input_bands_nums, out_channels=class_num).cuda()
    # model.load_state_dict(torch.load(r"/home/cjl/ywj_code/code/Multi-category_all/model_ori/4.pkl"))
    # criterion=nn.MSELoss()
    # criterion = nn.SmoothL1Loss()
    # criterion = nn.CrossEntropyLoss()
    criterion = OhemCrossEntropy(ignore_label=0,
                                 thres=OhemCrossEntropy_thres,
                                 min_kept=min_kept,
                                 weight=None,
                                 model_num_outputs=3,
                                 loss_balance_weights=[1, 1, 1])
    optimizer = torch.optim.Adam(model.parameters(), lr=mLearningRate)
    # optimizer = torch.optim.AdamW(model.parameters(),lr=mLearningRate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8, last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=12,
                                              verbose=True, threshold=0.005, threshold_mode='rel', cooldown=0,
                                              min_lr=0, eps=1e-08)
    # optim.lr_scheduler.ReduceLROnPlateau
    scaler = GradScaler()
    model.train()
    for epoch in range(mEpochs):
        # 训练
        # label_list = []
        # predict_list = []
        trainLossTotal = 0.0
        count_right = 0
        count_tot = 0
        for i, data in enumerate(trainLoader, 0):
            img, rgb_data, label = data #label 需要改成 0，1，2，3，4 的形式 4表示其他
            img, rgb_data, label = Variable(img).float().cuda(), Variable(rgb_data).float().cuda(), Variable(label).long().cuda()
            img = img.permute(0, 3, 1, 2)
            rgb_data = rgb_data / 255.0
            rgb_data -= mean
            rgb_data /= std
            rgb_data = rgb_data.permute(0, 3, 1, 2)
            with autocast():
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
            predictIndex = torch.argmax(predict[0], dim=1) #计算一下准确率和召回率 B*H*W 和label1一样
            # label[label == 0] = 255  # 255表示无标签位置
            # label1[label1 == class_num] = 0  # 0表示其他类别位置
            # label1 = label1.long()
            # tlabel = label1.clone()
            count_right += torch.sum((predictIndex == label) & (label != 0)).item()
            count_tot += torch.sum(label != 0).item()
            # #  label2=0 的地方 是未标注区域，直接赋值为真实标签，相当于不训练！！！
            # predict = torch.where(label2 == 0, one_hot, predict)  # label中没有标签的位置直接预测为真实值，也就相当于不参与loss的计算,而第四类转成了0，此时也会计算
            # # criterion 为 CrossEntropyLoss
            # loss = criterion(predict, label)

            # criterion 为 SmoothL1Loss
            # loss = criterion(predict, one_hot)

            # loss = focal_dice_loss(predict, one_hot,tlabel,0.5,weight_fun = 3,exp=10) #0,1,2,3,255
            # loss = focus_loss(class_num,predict, label)
            # trainLossTotal += loss.item()
            # optimizer.zero_grad()
            # loss.backward()
            # # optimizer.step()
            # scaler.step(optimizer)
            # scaler.update()

        accuracy = count_right / count_tot
        print('total train_loss = %.5f' % float(trainLossTotal))
        print('train epoch:', epoch, ': ', accuracy)
        scheduler.step(accuracy)
        print('learning rate = ', optimizer.state_dict()['param_groups'][0]['lr'])
        # label_list = np.array(label_list).flatten()
        # predict_list = np.array(predict_list).flatten()
        # print('label_shape: ',label_list.shape)
        # print('predict_shape: ', predict_list.shape)
        # ind = (label_list != 255)
        # predict_list = predict_list[ind]
        # label_list = label_list[ind]
        # print('train_epoch ', epoch ,":")
        # micro = f1_score(label_list, predict_list, average="micro")
        # macro = f1_score(label_list, predict_list, average="macro")

        # print('train_accuracy=', accuracy, 'train_micro=', micro, 'train_macro=', macro)
        # scheduler.step(accuracy)
        # torch.save(model.state_dict(), "model/concat11/params" + str(epoch) + ".pkl")
        # 测试
        # count_right = 0
        # count_tot = 0
        # testLossTotal = 0
        # # label_list = []
        # # predict_list = []
        # for i, data in enumerate(testLoader, 0):
        #     img, label = data
        #     label = label[:, 5:-5, 5:-5]
        #     # img, label = torch.tensor(img).float().cuda(), torch.tensor(label).float().cuda()
        #     img, label = Variable(img).float().cuda(), Variable(label).float().cuda()
        #     # tmp = torch.Tensor(img).to(mDevice)
        #     label1 = label.clone()  # label要先扩充批量维度 B*H*W,不用扩充，本来就是Batch
        #     label2 = torch.stack([label, label, label, label], 3)
        #     label2 = label2.permute(0, 3, 1, 2)  # B*C*H*W,
        #
        #     mask = torch.zeros(label.size()).to(mDevice)  # B*H*W
        #     label = torch.where(label == 4, mask, label)  # 第四类位置标0，0就代表其他类别 ，网络预测出0就表示其他类别
        #     label = label.long()  # index类型需转为整型
        #     one_hot = torch.nn.functional.one_hot(label, 4)  # 4分类
        #     one_hot = one_hot.float()
        #
        #     # print('type:',type(one_hot))
        #     # print('one_hot:',one_hot.size())
        #     one_hot = one_hot.permute(0, 3, 1, 2)  # B * C * H * W,
        #
        #     # print('one_hot:',one_hot.size())
        #     img = img.permute(0, 3, 1, 2)
        #     with torch.no_grad():
        #         predict = model(img)  # 只预测前三个通道
        #         # predict=model(img)
        #         predictIndex = torch.argmax(predict, dim=1)  # 计算一下准确率和召回率
        #         label1[label1 == 0] = 255  # 255表示无标签位置
        #         label1[label1 == 4] = 0  # 0表示其他类别位置
        #         label1 = label1.long()
        #
        #         count_right += torch.sum((predictIndex == label1) & (label1 != 255)).item()
        #         count_tot += torch.sum(label1 != 255).item()
        #
        #         # predictIndex = predictIndex.cpu().detach().numpy()
        #         # label1 = label1.cpu().detach().numpy()
        #
        #         # label_list.append(label1)
        #         # predict_list.append(predictIndex)
        #
        #
        #
        #         predict = torch.where(label2 == 0, one_hot, predict)
        #         loss = criterion(predict, one_hot)
        #         # loss = calc_loss(predict, one_hot)
        #         testLossTotal += loss.item()

            # testTotal += label.size(0)
            # predict = model(img)
            #
            # predict = torch.squeeze(predict)
            # # 计算正确率
            # predictIndex = torch.argmax(predict, dim=1)
            # labelIndex = torch.argmax(label, dim=1)
            # testCorrect += (predictIndex == labelIndex).sum()
        # print('test epoch:', epoch, ': ', count_right / count_tot)
        # print('total test_loss = %.5f' % float(testLossTotal))
        # label_list = np.array(label_list).flatten()
        # predict_list = np.array(predict_list).flatten()
        # ind = (label_list != 255)
        # predict_list = predict_list[ind]
        # label_list = label_list[ind]
        # print('epoch test',epoch,":")
        # micro = f1_score(label_list, predict_list, average="micro")
        # macro = f1_score(label_list, predict_list, average="macro")
        # accuracy = count_right / count_tot
        # print('test_accuracy=', accuracy, 'test_micro=', micro, 'test_macro', macro)
        # print('\n')
        # print('current lr: ',model.optimizer.state_dict()['param_groups'][0]['lr'])
        # print('learning rate = ', optimizer.state_dict()['param_groups'][0]['lr'])
        # scheduler.step()
        # scheduler.step(accuracy)
        # if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), model_path + str(epoch) + '.pkl')
