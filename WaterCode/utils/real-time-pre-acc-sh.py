import numpy as np
import os
import cv2
import csv
from tqdm import tqdm
# from data.dictNew import testFile
# import xlwt

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

weight = [0,1,1,1]
MUTI_CLASS_SH = [255 ,2,2,2,0,2,2,2,1,0,0,0,0,0,0,3,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0]

def transformLabel(gt, transform_rule):
    ind_list = []
    result = np.zeros(gt.shape, np.uint8)
    class_nums = len(transform_rule)
    for class_ind in range(class_nums):
        ind_list.append(np.where(gt == class_ind))
    for class_ind in range(class_nums):
        result[ind_list[class_ind]] = transform_rule[class_ind]
    return result

if __name__ == "__main__":
    gt_path = r"E:\DFX_result\label"+'\\'
    # pre_path = r"E:\DFX_result\shanghaishujuji\predict_label"+'\\'
    pre_path = r"E:\DFX_result\sh_testFileLabel\40" + '\\'
    # testfile = ['20210719134556845']
    testfile = ['20210720100823122']

    log = './log/'
    mkdir(log)
    csv2_save_path = log + 'test_statistics.csv'

    f2 = open(csv2_save_path, 'w', newline='')
    f2_csv = csv.writer(f2)
    csv2_header = ['file name','acc/%']
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
        if gt[:-4] not in testfile:
            continue
        gt_png = cv2.imread(gt_path+gt,cv2.IMREAD_GRAYSCALE)
        gt_png = gt_png[5:-5,5:-5]
        pre_png = cv2.imread(pre_path+pre,cv2.IMREAD_GRAYSCALE)
        gt_png = transformLabel(gt_png, MUTI_CLASS_SH)

        gt_list.append(gt_png)
        pre_list.append(pre_png)

        single_ind = (gt_png != 255)
        single_predict = pre_png[single_ind].flatten()
        single_label = gt_png[single_ind].flatten()
        single_iou = np.sum(single_predict == single_label)
        single_all = np.sum(single_predict == 1)  + np.sum(single_predict == 2)  + \
               np.sum(single_predict == 3) + np.sum(single_predict == 0)
        single_acc = single_iou/single_all
        single_acc_str = '%.3f' % (single_acc*100) +' ('+ str(single_iou) +'/' + str(single_all)+')'
        tmp_csv = [gt[:-4]+"\t",single_acc_str]
        f2_csv.writerow(tmp_csv)

    label_list = np.array(gt_list).flatten()
    predict_list = np.array(pre_list).flatten()
    ind = (label_list != 255)
    predict_list = predict_list[ind]
    label_list = label_list[ind]


    iou2 = np.sum(label_list == predict_list)
    all_cnts2 = np.sum(predict_list == 1)  + np.sum(predict_list == 2)  + \
               np.sum(predict_list == 3) + np.sum(predict_list == 0)

    acc2 = iou2 / all_cnts2
    print("20210720100823122 acc :", acc2)

    all_acc = ["20210720100823122 acc :", '%.3f' % (acc2*100) +' ('+ str(iou2) +'/' + str(all_cnts2)+')']
    f2_csv.writerow(all_acc)

    f2.close()