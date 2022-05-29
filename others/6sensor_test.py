from model_block.materialNet import *
import gc
from utils.os_helper import *
import csv
import copy
from utils.add_color import mask_color_img
from utils.accuracy_helper import *
from sklearn.metrics import f1_score
import sys

# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
# CUDA:0


FOR_TESTSET = int(sys.argv[1])
train_str = {0:'_train_',1:'_test_'}
png_path = '/home/cjl/data/sensor6/rgb/'
#csv2_save_path = './log/test_lr-4.csv'
csv2_save_path = './log/6sensor'+train_str[FOR_TESTSET]+'lr-4.csv'
LEN = 5
color_class = [[0,0,255],[255,0,0],[0,255,0]]
epoch_list = [str(i) for i in range(300)]



f2 = open(csv2_save_path, 'w', newline='')
f2_csv = csv.writer(f2)
csv2_header = ['epoch', 'accuracy', 'micro F1-score', 'macro F1-score']
f2_csv.writerow(csv2_header)


for epoch in epoch_list:
# MaterialModel input size in training = (m, channels, length, length)  (length == 11)
    model = MaterialSubModel(6, 4).cuda()
    model.load_state_dict(torch.load('/home/cjl/ywj_code/code/ori_multi-category/6sensor_model/' + epoch + '.pkl'))

    label_list = []
    predict_list = []
    count_right = 0
    count_tot = 0

    if FOR_TESTSET == 1:
        file_list = testFile_6se
        result_dir = './output_6sensor/' + epoch + '/test_lr-4/'
    else:
        file_list = trainFile_6se
        result_dir = './output_6sensor/' + epoch + '/train_lr-4/'
    mkdir(result_dir)
    for filename in file_list:
        img0 = cv2.imread(tif_dir + filename + '0.tif', -1)
        img1 = cv2.imread(tif_dir + filename + '1.tif', -1)
        img2 = cv2.imread(tif_dir + filename + '2.tif', -1)
        img3 = cv2.imread(tif_dir + filename + '3.tif', -1)
        img4 = cv2.imread(tif_dir + filename + '4.tif', -1)
        img5 = cv2.imread(tif_dir + filename + '5.tif', -1)
        # print(img5.shape, img2.dtype)
        mat_data = np.stack([img0, img1, img2, img3, img4, img5], axis=2)
        mat_data = mat_data.astype(np.float64)
        data_max = np.max(mat_data, axis=2, keepdims=True)
        imgData = mat_data / data_max
        #print(filename)
        # imgData = envi_loader(env_data_dir, filename)
        # imgData = transform2(imgData)
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

        imgGt = cv2.imread(png_path + filename + '.png')
        imgGt = imgGt[5:-5, 5:-5]

        imgRes = copy.deepcopy(imgGt)
        for color_num in [1, 2, 3]:
            imgRes = mask_color_img(imgRes, mask=(predict_ind == color_num), color=color_class[color_num - 1],
                                    alpha=0.7)
        # imgRes = np.zeros((rows, cols, 3), np.uint8)
        # imgRes[predict_ind == 1, 2] = 255
        # imgRes[predict_ind == 2, 0] = 255
        # imgRes[predict_ind == 3, 1] = 255

        cv2.imwrite(result_dir + filename + '.jpg', imgRes)

        img_label = cv2.imread(label_data_dir_6se + filename + '.png', cv2.IMREAD_GRAYSCALE)
        # img_label = img_label[5:5+rows, 5:5+cols]
        img_label = img_label[5:-5, 5:-5]
        # img_label = transformLabel(img_label, MUTI_CLASS)

        # imgGt = cv2.imread(png_path+filename+'.png')
        # imgGt = np.zeros((rows, cols, 3), np.uint8)
        # imgGt = imgGt1
        for color_num in [1, 2, 3]:
            imgGt = mask_color_img(imgGt, mask=(img_label == color_num), color=color_class[color_num - 1], alpha=0.7)
            # label_png = mask_color_img(label_png, label, color=class_color[i], alpha=0.7)
        # imgGt[img_label == 1, 2] = 255
        # imgGt[img_label == 2, 0] = 255
        # imgGt[img_label == 3, 1] = 255

        cv2.imwrite(result_dir + filename + '_gt.jpg', imgGt)
        # imgRes = np.zeros((rows, cols, 3), np.uint8)
        #
        # imgRes[predict_ind == 1, 2] = 255
        # imgRes[predict_ind == 2, 0] = 255
        # imgRes[predict_ind == 3, 1] = 255
        # cv2.imwrite(result_dir + filename + '.jpg', imgRes)
        #
        # img_label = cv2.imread(label_data_dir_6se + filename + '.png', cv2.IMREAD_GRAYSCALE)
        # img_label = img_label[5:5+rows, 5:5+cols]
        # # img_label = transformLabel(img_label, MUTI_CLASS)
        # imgGt = np.zeros((rows, cols, 3), np.uint8)
        # imgGt[img_label == 1, 2] = 255
        # imgGt[img_label == 2, 0] = 255
        # imgGt[img_label == 3, 1] = 255
        # cv2.imwrite(result_dir + filename + '_gt.jpg', imgGt)

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
