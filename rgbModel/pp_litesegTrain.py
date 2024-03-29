import sys
sys.path.append('../')
from model_block.materialNet import MaterialSubModel
from model_block.PP_liteseg_final import PPLiteSeg
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

waterImgRootPath = '/home/cjl/ssd/dataset/hefei/img/'
waterLabelPath = '/home/cjl/ssd/dataset/hefei/label/'
png_path = '/home/cjl/ssd/dataset/hefei/needtrain/'
hz_label = '/home/cjl/ssd/dataset/hangzhou/label/'
hz_png_path = '/home/cjl/ssd/dataset/hangzhou/rgb/'
bmh_label = '/home/cjl/dataset_18ch/WaterLabel_mask_221011/'
bmh_png_path = '/home/cjl/dataset_18ch/waterBmh/'
bmh_raw_path = '/home/cjl/dataset_18ch/waterBmh/'
skinCloth_raw_path = '/home/cjl/dataset_18ch/raw_data/'
skinCloth_raw_test_path = '/home/cjl/dataset_18ch/test_raw_data/'
skinCloth_label = '/home/cjl/dataset_18ch/label/'
# bmh_png_path =
select_train_bands = [114, 109, 125,  53, 108,  81, 100, 112,  25,  90,  96, 123 ]
input_bands_nums = len(select_train_bands)
nora = True
featureTrans = False
class_num = 3
# 这个参数最好大一些！！！
min_kept = args.min_kept
OhemCrossEntropy_thres = 0.9
num_classes = 3
input_channel = 3 # skin cloth other
mean = torch.tensor([0.5, 0.5, 0.5]).cuda()
std = torch.tensor([0.5, 0.5, 0.5]).cuda()

model_path = './model/PPLiteSeg_sinkCloth' + str(min_kept) + '_' + str(mLearningRate) + '_' + str(mtrainBatchSize) + '/'

print('mBatchSize',mtrainBatchSize)
print('mEpochs',mEpochs)
print('mLearningRate',mLearningRate)
# print('nora',nora)
# print('featureTrans', featureTrans)
print('min_kept', min_kept)

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

    trainDataset = Dataset_RGB(skinClothTrainFile, skinCloth_raw_path, skinCloth_label)
    testDataset = Dataset_RGB(skinClothTestFile, skinCloth_raw_path, skinCloth_label)
    trainLoader = DataLoader(dataset=trainDataset, batch_size=mtrainBatchSize, shuffle=True, drop_last=True)
    testLoader = DataLoader(dataset=testDataset, batch_size=mtrainBatchSize, shuffle=True, drop_last=True)
    model = PPLiteSeg(num_classes = num_classes, input_channel=input_channel).cuda()
    # model = MaterialSubModel(in_channels=input_bands_nums, out_channels=class_num).cuda()
    # model.load_state_dict(torch.load(r"/home/cjl/ywj_code/code/Multi-category_all/model_ori/4.pkl"))
    # criterion=nn.MSELoss()
    # criterion = nn.SmoothL1Loss()
    # criterion = nn.CrossEntropyLoss()
    criterion = OhemCrossEntropy(ignore_label=-1,
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
            img, label = data #label 需要改成 0，1，2，3，4 的形式 4表示其他
            img, label = Variable(img).float().cuda(), Variable(label).long().cuda()
            img = img / 255.0
            img -= mean
            img /= std
            img = img.permute(0, 3, 1, 2)
            # 其他类别映射成0
            pos_mask = (label != 1) & (label != 2)  # 相当于把没有标注的地方也算成了 other 也无所谓了！！
            label[pos_mask] = 0
            with autocast():
                predict = model(img)  # B*CLASS_NUM*H*W
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
            # predict = F.softmax(predict[0], dim=1)
            predictIndex = torch.argmax(predict[0], dim=1) #计算一下准确率和召回率 B*H*W 和label1一样
            count_right += torch.sum(predictIndex == label).item()  # 全部训练
            # count_right += torch.sum((predictIndex == label) & (label != 0)).item()
            count_tot += torch.sum(label != -1).item()
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
        count_right = 0
        count_tot = 0
        testLossTotal = 0
        # label_list = []
        # predict_list = []
        for i, data in enumerate(testLoader, 0):
            img, label = data  # label 需要改成 0，1，2，3，4 的形式 4表示其他
            img, label = Variable(img).float().cuda(), Variable(label).long().cuda()
            label[label==2] = 0 #后续只标注0类别
            img = img / 255.0
            img -= mean
            img /= std
            img = img.permute(0, 3, 1, 2)
            pos_mask = (label != 1) & (label != 2)
            label[pos_mask] = 0
            with torch.no_grad():
                predict = model(img)  # B*CLASS_NUM*H*W
            losses = criterion(predict, label)
            torch.unsqueeze(losses, 0)
            loss = losses.mean()
            predictIndex = torch.argmax(predict[0], dim=1)  # 计算一下准确率和召回率 B*H*W 和label1一样
            count_right += torch.sum(predictIndex == label).item()
            # count_right += torch.sum((predictIndex == label) & (label != 0)).item()
            count_tot += torch.sum(label != -1).item()
            testLossTotal += loss.item()
        print('test epoch:', epoch, ': ', count_right / count_tot)
        print('total test_loss = %.5f' % float(testLossTotal))
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
