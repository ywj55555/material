import numpy as np
import os
import cv2
from tqdm import tqdm
import csv

def mkdir(path):
    path = path.strip()
    path = path.rstrip('\\')
    path = path.rstrip('/')
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path,exist_ok=True)
        print("create dir(" + path + ") successfully")
        return True
    else:
        print("dir(" + path + ") is exist")
        return False
if __name__ == "__main__":
    gt_path =  r'E:\DFX_result\xitongceshiyongli\groundTruth'+'\\'
    pre_path = r'E:\DFX_result\xitongceshiyongli\predictLabel'+'\\'

    log = './log/'
    mkdir(log)
    csv2_save_path = log + 'system_statistics.csv'

    # testfile = ['20220111113416']
    testfile = ['20220111113419']

    f2 = open(csv2_save_path, 'w', newline='')
    f2_csv = csv.writer(f2)
    csv2_header = ['file name', 'acc/%']
    f2_csv.writerow(csv2_header)

    color_class = [[0, 0, 255], [255, 0, 0], [0, 255, 0]]

    gt_list = []
    pre_list = []

    gt_files = os.listdir(gt_path)
    gt_files = [ x for x in gt_files if x[-4:]=='.png']

    pre_files = os.listdir(pre_path)
    pre_files = [x for x in pre_files if x[-4:] == '.png' and x in gt_files]

    #计算acc
    for gt,pre in tqdm(zip(gt_files,pre_files)):
        # if gt[:-4] not in testfile:
        #     continue
        gt_png = cv2.imread(gt_path+gt,cv2.IMREAD_GRAYSCALE)
        pre_png = cv2.imread(pre_path+pre,cv2.IMREAD_GRAYSCALE)
        gt_png[gt_png==0]=255
        gt_png[gt_png == 3] = 0
        gt_list.append(gt_png)
        pre_list.append(pre_png)

        single_ind = (gt_png != 255)
        single_predict = pre_png[single_ind].flatten()
        single_label = gt_png[single_ind].flatten()
        single_iou = np.sum(np.logical_and(single_predict == single_label, single_label != 0))
        single_all = np.sum(single_predict == 1) + np.sum(single_predict == 2)

        single_acc = single_iou / single_all
        single_acc_str = '%.3f' % (single_acc * 100) + ' (' + str(single_iou) + '/' + str(single_all) + ')'
        tmp_csv = [gt[:-4] + "\t", single_acc_str]
        f2_csv.writerow(tmp_csv)

    label_list = np.array(gt_list).flatten()
    predict_list = np.array(pre_list).flatten()
    ind = (label_list != 255)
    predict_list = predict_list[ind]
    label_list = label_list[ind]

    iou = np.sum(np.logical_and(label_list == predict_list, label_list != 0))
    all_cnts = np.sum(predict_list == 1) + np.sum(predict_list == 2)
    acc = iou / all_cnts
    print("all:", acc)

    all_acc = ["all acc :", '%.3f' % (acc * 100) + ' (' + str(iou) + '/' + str(all_cnts) + ')']
    f2_csv.writerow(all_acc)

    f2.close()