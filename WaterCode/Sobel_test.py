import sys
sys.path.append('../')
import gc
import os
import torch.nn.functional as F
import torch
from torch import nn
import numpy as np
import spectral.io.envi as envi
from torch.autograd import Variable

wavelength = [449.466,451.022,452.598,454.194,455.809,457.443,459.095,460.765,462.452,464.157,465.879,467.618,469.374,471.146,472.935,474.742,
476.564,478.404,480.261,482.136,484.027,485.937,487.865,489.811,491.776,493.76,495.763,497.787,499.832,501.898,503.985,506.095,508.228,510.384,
512.565,514.771,517.003,519.261,521.547,523.86,526.203,528.576,530.979,533.414,535.882,538.383,540.919,543.49,546.098,548.743,551.426,554.149,
556.912,559.718,562.565,565.457,568.393,571.375,574.405,577.482,580.609,583.786,587.015,590.296,593.631,597.021,600.467,603.97,607.532,611.153,614.835,
618.578,622.385,626.255,630.19,634.192,638.261,642.399,646.606,650.883,655.232,659.654,664.15,668.72,673.367,678.09,682.891,687.771,692.73,697.77,702.892,
708.096,713.384,718.755,724.212,729.755,735.384,741.101,746.906,752.8,758.784,764.857,771.022,777.278,783.626,790.067,796.6,803.228,809.949,816.764,823.675,
830.68,837.781,844.978,852.271,859.659,867.144,874.725,882.403,890.176,898.047,906.013,914.076,922.235,930.49,938.84,947.285,955.825]

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

def envi_loader(dirpath, filename):
    judegHdrDataType(dirpath, filename)
    enviData = envi.open(dirpath + filename + '.hdr', dirpath + filename + '.img')
    imgData = enviData.load()
    imgData = np.array(imgData)
    gc.collect()
    return imgData

class Conv_3d_Sobel(nn.Module):
    """ Channel attention module"""
    def __init__(self):
        super(Conv_3d_Sobel, self).__init__()
        # self.sobel_kernel = Variable(torch.from_numpy(np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32').reshape((1, 1, 1, 3, 3))), requires_grad=False)
        sobel_kernel_0 = Variable(torch.from_numpy(np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype='float32').reshape((1, 1, 1, 3, 3))),requires_grad = False)
        sobel_kernel_1 = Variable(torch.from_numpy(np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]], dtype='float32').reshape((1, 1, 1, 3, 3))),requires_grad = False)
        sobel_kernel_2 = Variable(torch.from_numpy(np.array([[0, 0, -1], [0, 1, 0], [0, 0, 0]], dtype='float32').reshape((1, 1, 1, 3, 3))),requires_grad = False)
        sobel_kernel_3 = Variable(torch.from_numpy(np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]], dtype='float32').reshape((1, 1, 1, 3, 3))),requires_grad = False)
        sobel_kernel_4 = Variable(torch.from_numpy(np.array([[0, 0, 0], [0, 1, -1], [0, 0, 0]], dtype='float32').reshape((1, 1, 1, 3, 3))),requires_grad = False)
        sobel_kernel_5 = Variable(torch.from_numpy(np.array([[0, 0, 0], [0, 1, 0], [-1, 0, 0]], dtype='float32').reshape((1, 1, 1, 3, 3))),requires_grad = False)
        sobel_kernel_6 = Variable(torch.from_numpy(np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0]], dtype='float32').reshape((1, 1, 1, 3, 3))),requires_grad = False)
        sobel_kernel_7 = Variable(torch.from_numpy(np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]], dtype='float32').reshape((1, 1, 1, 3, 3))),requires_grad = False)
        self.sobel_kernel = torch.cat([sobel_kernel_0,sobel_kernel_1,sobel_kernel_2,sobel_kernel_3,sobel_kernel_4,sobel_kernel_5,sobel_kernel_6,sobel_kernel_7],dim = 0)

    def forward(self,x):
        # m_batchsize, C, height, width = x.size()
        x = x.unsqueeze(1)
        x = F.conv3d(x, self.sobel_kernel, stride=1, padding=(0,1,1)).abs().sum(1).squeeze()#C,height,width
        return x

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

def Save_raw_data(data, dir, name):
    channels = 128
    width = 1859
    height = 1415
    data = data.transpose(1,2,0)
    data = data.flatten()
    data.tofile(dir + name + '.raw')
    with open(dir + name+'.hdr','w')as f:
        f.write("ENVI"+"\n")
        f.write("samples = " + str(width)+"\n")
        f.write("lines = " + str(height)+"\n")
        f.write("bands = " + str(channels)+"\n")
        f.write("file type = ENVI Standard" + "\n")
        f.write("data type = 2" + "\n")
        f.write("interleave = bip" + "\n")
        f.write("sensor type = Unknown" + "\n")
        f.write("wavelength units = nm" + "\n")
        # f.write("header offset = 0" + "\n")
        # f.write("sensor type = CPDemo_8Channels" + "\n")
        f.write("wavelength = {")
        for i in range(127):
            # print(i)
            # print(wavelength[i])
            f.write(str(wavelength[i])+',')
        f.write(str(wavelength[127]) + '}\n')

#####################
data_dir = '/home/glk/datasets/hangzhou/'
file_dir = '20220318124159745'
data_dir_out = data_dir[:-1] + '_out/'
#####################

if __name__ == '__main__':
    file_num = 1
    model = Conv_3d_Sobel()
    mkdir(data_dir_out)
    for i in range(file_num):
        if os.path.exists(data_dir + file_dir + '.img'):
            print(data_dir + file_dir + '.img')
            imgData = envi_loader(data_dir, file_dir)
            inputData = torch.tensor(imgData).float().permute(2,0,1).unsqueeze(0)
            print(inputData.shape)
            output = model(inputData).detach().numpy()
            # out_file = 'out_raw_' + str(i)
            out_file = file_dir
            Save_raw_data(output,data_dir_out,out_file)

