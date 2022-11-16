import numpy as np
import spectral.io.envi as envi
import cv2
import gc
import os
import torch
import numpy as np
from torch import nn
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F

def nn_conv2d(im):
    conv_op = nn.Conv2d(1, 1, 3, bias=False)
    # sobel_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float32')#Gx
    # sobel_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype='float32')#Gy
    sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')#Sobel
    # sobel_kernel = np.array([[1, 4, 1], [4, -20, 4], [1, 4, 1]], dtype='float32')#Laplacian
    # sobel_kernel = np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]], dtype='float32')#Banner

    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    conv_op.weight.data = torch.from_numpy(sobel_kernel)

    edge_detect = conv_op(Variable(im))
    edge_detect = edge_detect.abs()#.pow(2)#Banner
    edge_detect = edge_detect.squeeze().detach().numpy()
    return edge_detect

def nn_conv3d(im):
    conv_op = nn.Conv3d(1, 2, (1,3,3), bias=False,padding = (0,1,1))
    sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')#Sobel
    # sobel_kernel = np.array([[1, 4, 1], [4, -20, 4], [1, 4, 1]], dtype='float32')#Laplacian
    # sobel_kernel = np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]], dtype='float32')#Banner

    sobel_kernel = sobel_kernel.reshape((1, 1, 1, 3, 3))
    print(conv_op.weight.data.shape)
    conv_op.weight.data = Variable(torch.from_numpy(sobel_kernel),requires_grad = False)
    edge_detect = conv_op(Variable(im))
    edge_detect = edge_detect.abs()#.pow(2)#Banner
    edge_detect = edge_detect.squeeze().detach().numpy()
    return edge_detect

def SMD(im):
    conv_op_1 = nn.Conv2d(1, 1, 3, bias=False)
    conv_op_2 = nn.Conv2d(1, 1, 3, bias=False)
    sobel_kernel_1 = np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]], dtype='float32')#Laplacian
    # sobel_kernel_1 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')#Laplacian
    sobel_kernel_1 = sobel_kernel_1.reshape((1, 1, 3, 3))
    conv_op_1.weight.data = torch.from_numpy(sobel_kernel_1)
    sobel_kernel_2 = np.array([[0, 0, 0], [0, 1, -1], [0, 0, 0]], dtype='float32')#Laplacian
    # sobel_kernel_2 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype='float32')#Laplacian
    sobel_kernel_2 = sobel_kernel_2.reshape((1, 1, 3, 3))
    conv_op_2.weight.data = torch.from_numpy(sobel_kernel_2)
    edge_detect_1 = conv_op_1(Variable(im)).abs()#.pow(2)
    edge_detect_2 = conv_op_2(Variable(im)).abs()#.pow(2)
    edge_detect = edge_detect_1 + edge_detect_2
    edge_detect = edge_detect.squeeze().detach().numpy()
    return edge_detect

def EAV(im):
    conv_op_1 = nn.Conv2d(1, 1, 3, bias=False)
    conv_op_2 = nn.Conv2d(1, 1, 3, bias=False)
    sobel_kernel_1 = np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]], dtype='float32')#x
    # sobel_kernel_1 = np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]], dtype='float32')#y
    sobel_kernel_1 = sobel_kernel_1.reshape((1, 1, 3, 3))
    conv_op_1.weight.data = torch.from_numpy(sobel_kernel_1)
    sobel_kernel_2 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype='float32')#Laplacian
    sobel_kernel_2 = sobel_kernel_2.reshape((1, 1, 3, 3))
    conv_op_2.weight.data = torch.from_numpy(sobel_kernel_2)
    edge_detect_1 = conv_op_1(Variable(im)).abs()#.pow(2)
    edge_detect = conv_op_2(edge_detect_1).abs()#.pow(2)
    edge_detect = edge_detect.squeeze().detach().numpy()
    return edge_detect

def envi_normalize(imgData):
    img_max =np.max(imgData,keepdims = True)
    return imgData / (img_max+0.0001)

def judegHdrDataType(hdr_dirpath, file):
    with open(hdr_dirpath +"/"+ file + '.hdr', "r") as f:
        data = f.readlines()
    modify_flag = False
    if data[5] != 'data type = 12\n':
        data[5] = 'data type = 12\n'
        modify_flag = True
        # raise HdrDataTypeError("data type = 2, but data type should be 12")
    if data[6].split(' =')[0] != 'byte order':
        data.insert(6,'byte order = 0\n')
        modify_flag = True
    else:
        if data[6] != 'byte order = 0\n':
            data[6] ='byte order = 0\n'
            modify_flag = True
    if modify_flag:
        with open(hdr_dirpath + "/" + file + '.hdr', "w") as f:
            f.writelines(data)
        print("mend the datatype of file : ", file)

def envi_loader(dirpath, filename,norma=True):
    judegHdrDataType(dirpath, filename)
    enviData = envi.open(dirpath + filename + '.hdr', dirpath + filename + '.img')
    imgData = enviData.load()
    imgData = np.array(imgData)
    imgData = envi_normalize(imgData)
    gc.collect()
    return imgData

def Low_Blur(img):
    # img_blur = cv2.blur(img,(5,5))
    img = np.float32(img)
    dft = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)  # 进行傅里叶变化
    dft_shift = np.fft.fftshift(dft)  # 将低频移到中间位置， 通常呈现中间亮，周围暗

    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)  # 中心位置
    # 低通滤波
    # mask = np.zeros((rows, cols, 2), np.uint8)  # 定义掩膜，中间黑，四周白。去除中间低频部分，即提取边界
    mask = np.ones((rows, cols, 2), np.uint8)
    T = 300
    a = 1
    mask[crow - T:crow + T, ccol - T:ccol + T] = a
    mask = mask + 1 - a

    # IDFT
    fshift = dft_shift * mask # 将掩模与傅里叶变化后图像相乘，保留中间部分
    f_ishift = np.fft.ifftshift(fshift)  # 将低频移动到原来的位置
    image_back = cv2.idft(f_ishift)  # 进行傅里叶的反变化
    image_back = cv2.magnitude(image_back[:, :, 0], image_back[:, :, 1])
    return image_back/image_back.max()*255

def Grad(img):
    tmp = img.copy()
    shape = np.shape(img)
    out = 0
    for x in range(1, shape[0] - 1):
        for y in range(1, shape[1] - 1):
            # grad
            # tmp[x, y] = abs(-1 * img[x-1,y-1] + -2 * img[x-1,y] + -1 * img[x-1,y+1] + img[x+1,y-1] + 2 * img[x+1,y] + img[x+1,y+1])#(**2) /16
            tmp[x, y] = abs(1 * img[x - 1, y - 1] + 2 * img[x, y - 1] + 1 * img[x + 1, y - 1] + -1 * img[x - 1, y + 1] + -2 *
                         img[x, y + 1] + -1 * img[x + 1, y + 1])# (** 2) / 16
    tmp = tmp / tmp.max() * 255
    return tmp

def Grad_torch(img):
    shape = np.shape(img)
    img = torch.from_numpy(img.reshape((1, 1, shape[0], shape[1])))
    img = img.abs()
    img = img / img.max() * 255
    img = nn_conv2d(img)
    # img = SMD(img)
    # img = EAV(img)
    return img

def Grad_torch_3D(img):
    shape = np.shape(img)
    img = torch.from_numpy(img.reshape((1, 1, shape[0], shape[1], shape[2]))).permute(0,1,4,2,3)
    img = img.abs()
    img = img / img.max() * 255
    img = nn_conv3d(img)
    return img

if __name__ == '__main__':
    dataset_dir = '/home/glk/datasets/hangzhou/'
    # dataset_dir = '/home/glk/datasets/hefei/All_data/'

    img = envi_loader(dataset_dir,'20220407163908168')
    W,H,C = img.shape
    img_list = []
    tmp_img = Grad_torch_3D(img)
    print(tmp_img.shape)
    for i in range(128):
        tmp = img[:,:,i]/img[:,:,i].max()*255
        # tmp = Grad(tmp)
        tmp = Grad_torch(tmp)
        print(tmp.shape)
        cv2.imwrite(dataset_dir + 'test/'+str(i)+'.jpg', tmp)
        img_list.append(dataset_dir + 'test/'+str(i)+'.jpg')
        if (i+1) % 8 == 0:
            cv2.imwrite(dataset_dir + 'test/' + str(i-7) + 'to' + str(i+1) + '.jpg',img[:,:,i-7:i+1].mean(2)/img[:,:,i-7:i+1].mean(2).max()*255)
    tmp = img[:,:,:].mean(2)/img[:,:,:].mean(2).max()
    cv2.imwrite(dataset_dir + 'test/mean.jpg',tmp * 255)
    cv2.imwrite(dataset_dir + 'test/mean_2.jpg', (img[:, :, :] / img[:, :, :].max(2, keepdims=True)).mean(2) * 255)  #
    img_tmp = img[:, :, :].mean(2) * 255
    print(img_tmp.mean())
    img_blur = Low_Blur(img_tmp)
    print(img_blur.mean())
    cv2.imwrite(dataset_dir + 'test/mean_blur.jpg', img_blur)
    Image = np.concatenate((img[:,:,:].mean(2,keepdims=True)*255,img[:,:,:90].mean(2,keepdims=True)*255,img[:,:,90:].mean(2,keepdims=True)*255),axis=2)
    cv2.imwrite(dataset_dir + 'test/feature.jpg',Image)
    # fps = 12
    # size=(H,W)
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # videoWriter = cv2.VideoWriter('./channel.mp4',fourcc,fps,size, True)
    # for frame_file in img_list:
    #     frame = cv2.imread(frame_file,flags =0)
    #     # cv2.imwrite('test.jpg',frame)
    #     frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    #     print(frame.shape)
    #     videoWriter.write(frame)
    # videoWriter.release()
