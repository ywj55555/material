import numpy as np
import cv2
import os
# a = np.array([[1,2,2],[2,1,3]])
# print(a[np.logical_or(a==1,a==2)])
from data.dictNew import testFile
def returnint():
    a = 5
    print("hanshu : ",id(a))
    return a
if __name__ == '__main__':
    pass
    # gt_path = r"E:\DFX_result\shanghaishujuji\gt" + '\\'
    # gt_files = os.listdir(gt_path)
    # for gt in gt_files:
    #     if gt[:-4] in testFile:
    #         print(gt[:-4])
    # labelCount = 300
    # class_nums = 3
    # per_class_num = [labelCount * 2]
    # per_class_num_care = [labelCount for _ in range(class_nums - 1)]
    # per_class_num.extend(per_class_num_care)

    # print(per_class_num)
    #     gt_png = cv2.imread(gt_path+gt,cv2.IMREAD_GRAYSCALE)
    #     print(gt_png.shape)
    #     gt_png2 = cv2.imread(gt_path + gt)
    #     print(gt_png2.shape)
    #     for i in range(3):
    #         print((gt_png2[:,:,i]==gt_png).all())
    #     break

    # for _ in range(10):
    #     # i = features_status[_]
    #     j = list(range(10))[_]
    #     print(_,j)
    # a = np.array([[1,2,3],[4,5,6]])
    # b = a
    # print(id(a))
    # print(id(b))
    #
    # c = a.copy()
    # print(id(c))
    # a[0]=100
    # print(a,"\n",b,"\n",c)
    # print(id(a))
    # print(id(b))
    # print(id(c))