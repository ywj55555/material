from model_block.materialNet import *
from utils.accuracy_helper import *
from utils.add_color import *
from utils.os_helper import mkdir


'''
net_cloth_dir = './model/cloth_4.pkl'
net_skin_dir = './model/skin_9.pkl'
net_plant_dir = './model/plant_9.pkl'
'''

'''
net_cloth_dir = './model/cloth_54.pkl'
net_skin_dir = './model/skin_59.pkl'
net_plant_dir = './model/plant_59.pkl'
'''

# FOR_TESTSET = 0
# csv2_save_path = './log_cut/test_lr-4.csv'
# # csv2_save_path = './log_cut/train_lr-4.csv'
# LEN = 5




# f2 = open(csv2_save_path, 'w', newline='')
# f2_csv = csv.writer(f2)
# csv2_header = ['epoch', 'accuracy', 'micro F1-score', 'macro F1-score']
# f2_csv.writerow(csv2_header)


# for epoch in epoch_list:
# MaterialModel input size in training = (m, channels, length, length)  (length == 11)
#     model = MaterialSubModel(20, 4).cuda()
#     model = MaterialSubModel(29, 4).cuda()
    # model.load_state_dict(torch.load('./model/lr-4/' + epoch + '.pkl'))
# env_data_dir = ''
color_class = [[0,0,255],[255,0,0],[0,255,0]]
png_path = "/home/cjl/data/png/"
filename = '20210329154859783'
result_dir ="model_test_file/"
npy_path = "/home/cjl/data/npy_trans/envi/"
mkdir(result_dir)
model_path ="/home/cjl/ywj_code/code/ori_multi-category/ddp_model/36.ckpt"
# model = MaterialSubModel(in_channels=20, out_channels=4, kernel_size = 7,padding_mode='reflect',mid_channels_1=40, mid_channels_2=60, mid_channels_3=16).cuda()
# model = MaterialSubModel(in_channels=20, out_channels=4, kernel_size = 5,padding_mode='reflect',mid_channels_1=24, mid_channels_2=12, mid_channels_3=6).cuda()
model = MaterialSubModel(20, 4).cuda()
model.load_state_dict(torch.load(model_path))
    # model.load_state_dict(torch.load('./model_cut/' + epoch + '.pkl'))

    # label_list = []
    # predict_list = []
    # count_right = 0
    # count_tot = 0

    # if FOR_TESTSET == 1:
    #     file_list = testFile
    #     result_dir = './output_cut/' + epoch + '/test_lr-4/'
    # else:
    #     file_list = trainFile
    #     result_dir = './output_cut/' + epoch + '/train_lr-4/'
    # mkdir(result_dir)
    # for filename in file_list:
        #print(filename)

# imgData = envi_loader(env_data_dir, filename)
# imgData = transform2(imgData)
#         # imgData, imgData_cut = envi_loader_cut(env_data_dir, filename)
#         # imgData = transform_cut(imgData, imgData_cut)
# imgData = imgData.transpose(2, 0, 1)
imgData = np.load(npy_path+filename+'.npy')
imgData = np.expand_dims(imgData, 0)
inputData = torch.tensor(imgData).float().cuda()
with torch.no_grad():
    inputData = inputData.permute(0,3,1,2)
    predict = model(inputData)
    predict = torch.squeeze(predict)
    predict_ind = torch.argmax(predict, dim=0)
    predict_ind = predict_ind.cpu().detach().numpy()
    #print(np.unique(predict_ind))

    imgGt = cv2.imread(png_path + filename + '.png')
    # imgGt =
    imgRes = imgGt[5:-5,5:-5]
    for color_num in [1, 2, 3]:
        imgRes = mask_color_img(imgRes, mask=(predict_ind == color_num), color=color_class[color_num - 1], alpha=0.7)
    # imgRes = np.zeros((rows, cols, 3), np.uint8)
    # imgRes[predict_ind == 1, 2] = 255
    # imgRes[predict_ind == 2, 0] = 255
    # imgRes[predict_ind == 3, 1] = 255

    cv2.imwrite(result_dir + filename + '.jpg', imgRes)

    img_label = cv2.imread(label_data_dir + filename + '.png', cv2.IMREAD_GRAYSCALE)
    # img_label = img_label[5:5+rows, 5:5+cols]
    img_label = transformLabel(img_label, MUTI_CLASS)#0->255 other->0

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

    # generate result
    # rows, cols = predict_ind.shape
    # imgRes = np.zeros((rows, cols, 3), np.uint8)
    # imgRes[predict_ind == 1, 2] = 255
    # imgRes[predict_ind == 2, 0] = 255
    # imgRes[predict_ind == 3, 1] = 255
    # cv2.imwrite( filename + '.jpg', imgRes)
    #
    # img_label = cv2.imread(label_data_dir + filename + '.png', cv2.IMREAD_GRAYSCALE)
    # # img_label = img_label[5:5+rows, 5:5+cols]
    # img_label = transformLabel(img_label, MUTI_CLASS)
    # imgGt = np.zeros((rows, cols, 3), np.uint8)
    # imgGt[img_label == 1, 2] = 255
    # imgGt[img_label == 2, 0] = 255
    # imgGt[img_label == 3, 1] = 255
    # cv2.imwrite(filename + '_gt.jpg', imgGt)
    print('successful')

    # calculate TN/TF/F-score and so on
    # predict_0, gt_0 = countPixels(img_label, 0, predict_ind, 0) #预测正确数量、真实标签数量
    # predict_1, gt_1 = countPixels(img_label, 1, predict_ind, 1)
    # predict_2, gt_2 = countPixels(img_label, 2, predict_ind, 2)
    # predict_3, gt_3 = countPixels(img_label, 3, predict_ind, 3)

    # label_list.append(img_label)
    # predict_list.append(predict_ind)
    # count_right += (predict_0 + predict_1 + predict_2 + predict_3)
    # count_tot += (gt_0 + gt_1 + gt_2 + gt_3)

#     gc.collect()
#     label_list = np.array(label_list).flatten()
#     predict_list = np.array(predict_list).flatten()
#     ind = (label_list != 255)
#     predict_list = predict_list[ind]
#     label_list = label_list[ind]
#     print('epoch ' + epoch + ":")
#     micro = f1_score(label_list, predict_list, average="micro")
#     macro = f1_score(label_list, predict_list, average="macro")
#     accuracy = count_right / count_tot
#     print('accuracy=', accuracy, 'micro=', micro, 'macro', macro)
#     csv2_line = [epoch, accuracy, micro, macro]
#     f2_csv.writerow(csv2_line)
#     gc.collect()
# f2.close()
