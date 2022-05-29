'''
每张图生成2250个测试像素，间隔取样2000个点左右（标注的点）展开成一维取样？？二维间隔取样才能保证稀疏性，更加模拟整图预测
网格取样需要每一类单独计算间隔，然后按照每一类别数量在整图的比例进行取样
'''
import torch
from model_block import network
from utils.load_spectral import envi_loader
from utils.accuracy_helper import *
from utils.os_helper import mkdir
from utils.parse_args import parse_test_args
import os

envi_dir = '/home/cjl/dataset/envi/'
label_dir = '/home/cjl/dataset/label/'
png_dir = '/home/cjl/dataset/rgb/'

log = './log/'
mkdir(log)


det_num = 12000
train_mean = np.array([1638.30293365, 1856.07645628, 2395.36102414, 2963.61267338, 3825.72213097,
 3501.42091296, 4251.15954562, 3965.7991342,  3312.6616123 ])
# train_mean = np.array(train_mean)
train_var = np.array([ 5402022.45076405,  6679240.29294402,  9629853.43114098, 15170998.1931259,
 25767233.41870246, 18716092.38116236, 18231725.96313715, 28552394.38170321,
 13127436.03417886])
train_mean = torch.from_numpy(train_mean).float().cuda()
train_var = torch.from_numpy(train_var).float().cuda()
# train_var = np.array(train_var)

args = parse_test_args()
model_select = args.model_select
print('model select: ',model_select)
model_str = {0:'DBDA',1:'SSRN'}
model_list = ['/home/cjl/ywj_code/contrast_model_sh/DBDA_model_cut_scale32_0.0005/277.pkl',
              '/home/cjl/ywj_code/contrast_model_sh/SSRN_model_cut_scale32_0.0005/168.pkl']
bands = 9
CLASSES_NUM = 4
device = torch.device('cuda')

if model_select==0:
    model = network.DBDA_network_MISH(bands, CLASSES_NUM).to(device)
else:
    model = network.SSRN_network(bands, CLASSES_NUM).to(device)
model_path = model_list[model_select]
model.load_state_dict(torch.load(model_path))
model.eval()

# csv2_save_path = log+'pix_predict'+model_str[model_select]+'data.csv'
# f2 = open(csv2_save_path, 'w', newline='')
# f2_csv = csv.writer(f2)
# csv2_header = ['micro accuracy']
# f2_csv.writerow(csv2_header)


label_list = []
predict_list = []
with torch.no_grad():
    for file in testfile:
        label_data = cv2.imread(label_dir+file+'.png',cv2.IMREAD_GRAYSCALE)
        # 要先计算训练集的均值和方差！！
        imgdata = envi_loader(os.path.join(envi_dir,file[:8])+'/', file,False)

        #判断行列顺序是否相反
        if imgdata.shape[0]!=label_data.shape[0]:
            label_data = label_data.transpose(1,0)

        label_tmp = label_data[5:-5,5:-5]
        label_tmp = transformLabel(label_tmp, MUTI_CLASS_SH) #255,0,1,2,3
        label_fla = label_tmp.flatten()
        tmp_w = label_tmp.shape[0]
        tmp_h = label_tmp.shape[1]
        label_coor = [[x,y] for x in range(tmp_w) for y in range(tmp_h)]
        label_coor = np.array(label_coor)
        remain_index = label_fla!=255
        remain_pix = label_fla[remain_index]

        remain_coor = label_coor[remain_index]
        interval = remain_pix.shape[0]//det_num
        # det_pix = remain_pix[::interval]
        det_coor = remain_coor[::interval]
        remain_pix = remain_pix[::interval]
        label_list.extend(remain_pix)
        # print(det_coor.shape[0])
        # img_input = np.empty()
        pred_total = 0
        pred_correct = 0
        imgData = torch.from_numpy(imgdata).float().cuda()
        labelData = torch.from_numpy(remain_pix).long().cuda()

        pred_total+=labelData.size()[0]
        # imgdata = imgdata.float().cuda()
        imgData = imgData.unsqueeze(0)  # B C H W D
        imgData = imgData.unsqueeze(0)
        pix_data = torch.empty([0, 1, 11, 11, 9], dtype=torch.float).cuda()
        for coor in det_coor:
            pix_data = torch.cat((pix_data, imgData[:, :, coor[0] :coor[0]+ 11, coor[1]:coor[1] +11, :]), 0)
        # pix_data_ = pix_data.permute()
        pix_data_ = pix_data.view(np.prod(pix_data.shape[:4]),pix_data.shape[4])

        pix_data_ = (pix_data_-train_mean)/torch.sqrt(train_var)

        pix_data = pix_data_.view(pix_data.shape)
        predict = model(pix_data)
        # print(predict.shape[0])
        # predict = torch.squeeze(predict)
        predict_ind = torch.argmax(predict, dim=1)  # B：只有一个维度
        predict_ind = predict_ind.squeeze() #B
        pred_correct += (predict_ind == labelData).sum().item()
        predict_list.extend(predict_ind.cpu().numpy())
        print(file,' acc: ',pred_correct/pred_total)

label_list = np.array(label_list).flatten()
predict_list = np.array(predict_list).flatten()
acc_nums = np.sum(label_list==predict_list)
print("all acc:",acc_nums/len(label_list))

iou = np.sum(np.logical_and(label_list==predict_list,label_list!=0))
all_cnts = np.sum(label_list==1)+np.sum(label_list==2)+np.sum(label_list==3)
three_acc = iou/all_cnts
print("three acc:",three_acc)


# target_names = ['other','skin_','cloth','plant']
# res = classification_report(label_list, predict_list, target_names=target_names,output_dict=True)
#
# csv2_line = [res['accuracy'], res['macro avg']['f1-score']]
# # print('accuracy=', accuracy, 'micro=', micro, 'macro', macro)
# for k in target_names:
#     # print(k,'skin pre=', res['skin']['precision'], 'skin rec=', res['skin']['recall'])
#     print(k, ' pre=', res[k]['precision'], ' rec=', res[k]['recall'], ' f1-score=', res[k]['f1-score'])
#     csv2_line.extend([res[k]['precision'], res[k]['recall'], res[k]['f1-score']])
# print('all test micro accuracy:', res['accuracy'], ' all test macro avg f1:', res['macro avg']['f1-score'])
# f2_csv.writerow(csv2_line)
# f2.close()









