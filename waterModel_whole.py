from model_block.materialNet import MaterialSubModel
from utils.os_helper import mkdir
from utils.focalloss import focus_loss
from utils.cal_loss import focal_dice_loss
from model_block.Dataset import *
from data.dictNew import *
from torch.autograd import Variable
# os.environ['CUDA_VISIBLE_DEVICES'] = '5'
# CUDA:0

mtrainBatchSize = 2 #后期增加一个训练图变成38 整除2，或者其他
# mtestBatchSize = 2
mEpochs = 300
# start = 5
mLearningRate = 0.0001
# mLearningRate = 2.5e-05
mDevice=torch.device("cuda")

waterImgRootPath = '/home/cjl/ssd/dataset/shenzhen/img/train/'
waterLabelPath = '/home/cjl/ssd/dataset/shenzhen/label/Label_rename/'
select_train_bands = [114, 109, 125,  53, 108,  81, 100, 112,  25,  90,  96, 123 ]
input_bands_nums = len(select_train_bands)
nora = True
featureTrans = False
class_num = 2

model_path = './small_model_whole_' + str(mLearningRate) + '_' + str(mtrainBatchSize) + '_'\
             + str(nora) + '_' + str(featureTrans)+'/'

print('mBatchSize',mtrainBatchSize)
print('mEpochs',mEpochs)
print('mLearningRate',mLearningRate)
print('nora',nora)

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

    trainDataset = Dataset_all(SeaFile, waterImgRootPath, waterLabelPath, select_train_bands, nora, featureTrans)
    # testDataset = Dataset_all(testFile_hz,train_data)
    trainLoader = DataLoader(dataset=trainDataset, batch_size=mtrainBatchSize, shuffle=True)
    # testLoader = DataLoader(dataset=testDataset, batch_size=mtestBatchSize, shuffle=True)

    model = MaterialSubModel(in_channels=input_bands_nums, out_channels=class_num).cuda()
    # model.load_state_dict(torch.load(r"/home/cjl/ywj_code/code/Multi-category_all/model_ori/4.pkl"))
    # criterion=nn.MSELoss()
    # criterion = nn.SmoothL1Loss()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=mLearningRate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8, last_epoch=-1)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=12,
    #                                           verbose=True, threshold=0.005, threshold_mode='rel', cooldown=0,
    #                                           min_lr=0, eps=1e-08)
    # optim.lr_scheduler.ReduceLROnPlateau
    for epoch in range(mEpochs):

        # 训练
        # label_list = []
        # predict_list = []
        trainLossTotal = 0.0
        count_right = 0
        count_tot = 0
        for i, data in enumerate(trainLoader, 0):
            # print(i)
            img, label = data #label 需要改成 0，1，2，3，4 的形式 4表示其他
            label = label[:,5:-5,5:-5]
            img, label = Variable(img).float().cuda(), Variable(label).float().cuda()
            # img, label = torch.tensor(img).float().cuda(), torch.tensor(label).float().cuda()
            # trainTotal += label.size(0)
            # label1 = label  #B H W
            label1 = label.clone()

            # 用于保存原label中 0，1，2；0表示未标注，不参与训练
            label2 = torch.stack([label for _ in range(class_num)], 3)  # B H W*4
            # label2 = torch.stack([label, label, label, label], 3) #B H W*4
            label2 = label2.permute(0, 3, 1, 2)  # B*C*H*W,

            # 把 class_num 类别赋值为0：其他类别
            mask = torch.zeros(label.size()).to(mDevice)  # B*H*W
            label = torch.where(label == class_num, mask, label)  # 第class_num类标签为0，0就代表其他类别
            label = label.long()  # index类型需转为整型
            one_hot = torch.nn.functional.one_hot(label, class_num) #2分类
            one_hot = one_hot.float()
            # print('type:',type(one_hot))
            # print('one_hot:',one_hot.size())
            one_hot = one_hot.permute(0, 3, 1, 2)# B * C * H * W,
            # print('one_hot:',one_hot.size())
            img = img.permute(0, 3, 1, 2)
            predict = model(img)  # B*CLASS_NUM*H*W
            # predict=model(img)
            predictIndex = torch.argmax(predict, dim=1) #计算一下准确率和召回率 B*H*W 和label1一样
            label1[label1 == 0] = 255  # 255表示无标签位置
            label1[label1 == class_num] = 0  # 0表示其他类别位置
            label1 = label1.long()
            tlabel = label1.clone()
            count_right += torch.sum((predictIndex == label1) & (label1 != 255)).item()
            count_tot += torch.sum(label1 != 255).item()
            #  label2=0 的地方 是未标注区域，直接赋值为真实标签，相当于不训练！！！
            predict = torch.where(label2 == 0, one_hot, predict)  # label中没有标签的位置直接预测为真实值，也就相当于不参与loss的计算,而第四类转成了0，此时也会计算
            # criterion 为 CrossEntropyLoss
            loss = criterion(predict, label)

            # criterion 为 SmoothL1Loss
            # loss = criterion(predict, one_hot)

            # loss = focal_dice_loss(predict, one_hot,tlabel,0.5,weight_fun = 3,exp=10) #0,1,2,3,255
            # loss = focus_loss(class_num,predict, label)
            trainLossTotal += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # predict = model(img)
            # 计算正确率
            # predict = predict.squeeze()
            # predictIndex = torch.argmax(predict, dim=1)
            # labelIndex = torch.argmax(label, dim=1)
            # trainCorrect += (predictIndex == labelIndex).sum()
            # 产生loss
            # loss = criterion(predict, label)
            # trainLossTotal += loss
            # print("loss = %.5f" % float(loss))
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

        accuracy = count_right / count_tot
        print('total train_loss = %.5f' % float(trainLossTotal))
        print('train epoch:', epoch, ': ', accuracy)
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
