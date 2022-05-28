from materialNet import *
from data.utilNetNew import *
from utils.load_spectral import *
import cv2
import gc
from utils.os_helper import *
import csv
from utils.accuracy_helper import *
from sklearn.metrics import f1_score

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
# CUDA:0

FOR_TESTSET = 0
#csv2_save_path = './log/test_lr-4.csv'
csv2_save_path = './log/train_lr-4.csv'
LEN = 5

def predictIndex4image(imgData, typeCode, model):
    # imgData.shape = (rows, cols, channels), eg:(1409, 1692, 128)
    inputData = transform(imgData, typeCode)
    # inputData.shape = (rows, cols, now_channels), eg:(1409, 1692, 8)
    inputData = inputData.transpose(2, 0, 1)
    inputData = np.expand_dims(inputData, 0)
    # inputData.shape = (1, now_channels, rows, cols), eg:(1, 8, 1409, 1692)
    inputData = torch.tensor(inputData).float().cuda()
    predict = model(inputData)
    # predict.shape = (1, hot-code Size, now_rows, now_cols), eg:(1, 2, 1399, 1682)
    predict = torch.squeeze(predict)
    # predict.shape = (hot-code Size, now_rows, now_cols), eg:(2, 1399, 1682)
    predictIndex = torch.argmax(predict, dim=0)
    # predictIndex.shape = (now_rows, now_cols), eg:(1399, 1682)
    predictIndex = predictIndex.cpu().numpy()
    return predictIndex

def predict4image(imgData, typeCode, model):
    # imgData.shape = (rows, cols, channels), eg:(1409, 1692, 128)
    inputData = transform(imgData, typeCode)
    # inputData.shape = (rows, cols, now_channels), eg:(1409, 1692, 8)
    inputData = inputData.transpose(2, 0, 1)
    inputData = np.expand_dims(inputData, 0)
    # inputData.shape = (1, now_channels, rows, cols), eg:(1, 8, 1409, 1692)
    inputData = torch.tensor(inputData).float().cuda()
    predict = model(inputData)
    # predict.shape = (1, hot-code Size, now_rows, now_cols), eg:(1, 2, 1399, 1682)
    predict = torch.squeeze(predict)
    # predict.shape = (hot-code Size, now_rows, now_cols), eg:(2, 1399, 1682)
    predict = predict.cpu().detach().numpy()
    predict = predict.transpose(1, 2, 0)
    # predict.shape = (now_rows, now_cols, hot-code Size), eg:(1399, 1682, 2)
    return predict


epoch_list = [str((i + 1) * 10 - 1) for i in range(30)]

f2 = open(csv2_save_path, 'w', newline='')
f2_csv = csv.writer(f2)
csv2_header = ['epoch', 'accuracy', 'micro F1-score', 'macro F1-score']
f2_csv.writerow(csv2_header)


for epoch in epoch_list:
# MaterialModel input size in training = (m, channels, length, length)  (length == 11)
    model = MaterialSubModel(20, 4).cuda()
    model.load_state_dict(torch.load('./model/lr-4/' + epoch + '.pkl'))

    label_list = []
    predict_list = []
    count_right = 0
    count_tot = 0

    if FOR_TESTSET == 1:
        file_list = testFile
        result_dir = './output/' + epoch + '/test_lr-4/'
    else:
        file_list = trainFile
        result_dir = './output/' + epoch + '/train_lr-4/'
    mkdir(result_dir)
    for filename in file_list:
        #print(filename)
        imgData = envi_loader(env_data_dir, filename)
        imgData = transform2(imgData)
        imgData = imgData.transpose(2, 0, 1)
        imgData = np.expand_dims(imgData, 0)
        inputData = torch.tensor(imgData).float().cuda()

        predict = model(inputData)
        predict = torch.squeeze(predict)
        predict_ind = torch.argmax(predict, dim=0)
        predict_ind = predict_ind.cpu().detach().numpy()
        #print(np.unique(predict_ind))

        # generate result
        rows, cols = predict_ind.shape
        imgRes = np.zeros((rows, cols, 3), np.uint8)
        imgRes[predict_ind == 1, 2] = 255
        imgRes[predict_ind == 2, 0] = 255
        imgRes[predict_ind == 3, 1] = 255
        cv2.imwrite(result_dir + filename + '.jpg', imgRes)

        img_label = cv2.imread(label_data_dir + filename + '.png', cv2.IMREAD_GRAYSCALE)
        img_label = img_label[5:5+rows, 5:5+cols]
        img_label = transformLabel(img_label, MUTI_CLASS)
        imgGt = np.zeros((rows, cols, 3), np.uint8)
        imgGt[img_label == 1, 2] = 255
        imgGt[img_label == 2, 0] = 255
        imgGt[img_label == 3, 1] = 255
        cv2.imwrite(result_dir + filename + '_gt.jpg', imgGt)

        # calculate TN/TF/F-score and so on
        predict_0, gt_0 = countPixels(img_label, 0, predict_ind, 0)
        predict_1, gt_1 = countPixels(img_label, 1, predict_ind, 1)
        predict_2, gt_2 = countPixels(img_label, 2, predict_ind, 2)
        predict_3, gt_3 = countPixels(img_label, 3, predict_ind, 3)

        label_list.append(img_label)
        predict_list.append(predict_ind)
        count_right += (predict_0 + predict_1 + predict_2 + predict_3)
        count_tot += (gt_0 + gt_1 + gt_2 + gt_3)

        gc.collect()
    label_list = np.array(label_list).flatten()
    predict_list = np.array(predict_list).flatten()
    ind = (label_list != 255)
    predict_list = predict_list[ind]
    label_list = label_list[ind]
    print('epoch ' + epoch + ":")
    micro = f1_score(label_list, predict_list, average="micro")
    macro = f1_score(label_list, predict_list, average="macro")
    accuracy = count_right / count_tot
    print('accuracy=', accuracy, 'micro=', micro, 'macro', macro)
    csv2_line = [epoch, accuracy, micro, macro]
    f2_csv.writerow(csv2_line)
    gc.collect()
f2.close()
