import spectral.io.envi as envi
import os
import numpy as np
import gc
import matplotlib.pyplot as plt

dirpath = r'D:\dataset\baiban'+'\\'
qingtianImg = '20230214140155347'
qingtianPos = [1095, 678] # 先是横向坐标
qingtianyingyinImg = '20230214140327763'
qingtianyingyinPos = [1067, 573]
yintianImg = '20230207130739895'
yintianPos = [856, 829]
yintianyingyinImg = '20230207131503488'
yintianyingyinPos = [1006, 797]

filelist = [qingtianImg, qingtianyingyinImg, yintianImg, yintianyingyinImg]
posList = [qingtianPos, qingtianyingyinPos, yintianPos, yintianyingyinPos]
legendList = ['晴天', '晴天阴影', '阴天', '阴天阴影']

dirpath_sh = r'F:\shanghai-multispectral\envi'+'\\'
dirpath_hz = r'F:\hangzhou\envi'+'\\'
dirpath_water1 = r'E:\shenzhen\img\20220427'+'\\'
dirpath_water2 = r'E:\shenzhen\img\20220429'+'\\'
dirpath_water3 = r'E:\hefei\img'+'\\'
filepathlist = [dirpath_sh, dirpath_hz, dirpath_water1, dirpath_water2, dirpath_water3]

fileOrder = ['20210521101100292', '20210521101100292', '20210521101100292', '20210521101100292', '20210521095839946',
             '20210521095839946', '20210521095839946', '20210521100704811', '20220427144846394']
posOrder = [[687, 1098], [452, 221], [1090, 392], [1004, 1077], [701, 817], [662, 223], [1428, 1099], [496, 687], [1085, 899]]
objectOrder = ['皮肤', '浅蓝色牛仔衣', '银色汽车喷漆', '水泥路面', '暗色系棉质衣物', '绿植', '灰白色石砖', '灰色墙砖', '江河湖水']

def envi_normalize(imgData):
    img_max = np.max(imgData, axis=2, keepdims=True) #通道归一化
    # return imgData / img_max[:, :, np.newaxis]
    return imgData / img_max

def plot_embedding(X, y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    # ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]), color=plt.cm.Set1(y[i] / 10.), fontdict={'weight': 'bold', 'size': 9})

wavelength = [
449.466,
451.022,
452.598,
454.194,
455.809,
457.443,
459.095,
460.765,
462.452,
464.157,
465.879,
467.618,
469.374,
471.146,
472.935,
474.742,
476.564,
478.404,
480.261,
482.136,
484.027,
485.937,
487.865,
489.811,
491.776,
493.76,
495.763,
497.787,
499.832,
501.898,
503.985,
506.095,
508.228,
510.384,
512.565,
514.771,
517.003,
519.261,
521.547,
523.86,
526.203,
528.576,
530.979,
533.414,
535.882,
538.383,
540.919,
543.49,
546.098,
548.743,
551.426,
554.149,
556.912,
559.718,
562.565,
565.457,
568.393,
571.375,
574.405,
577.482,
580.609,
583.786,
587.015,
590.296,
593.631,
597.021,
600.467,
603.97,
607.532,
611.153,
614.835,
618.578,
622.385,
626.255,
630.19,
634.192,
638.261,
642.399,
646.606,
650.883,
655.232,
659.654,
664.15,
668.72,
673.367,
678.09,
682.891,
687.771,
692.73,
697.77,
702.892,
708.096,
713.384,
718.755,
724.212,
729.755,
735.384,
741.101,
746.906,
752.8,
758.784,
764.857,
771.022,
777.278,
783.626,
790.067,
796.6,
803.228,
809.949,
816.764,
823.675,
830.68,
837.781,
844.978,
852.271,
859.659,
867.144,
874.725,
882.403,
890.176,
898.047,
906.013,
914.076,
922.235,
930.49,
938.84,
947.285,
955.825
]

if __name__ == '__main__':
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 获取other的光谱特征曲线
    lastdata = None
    lastfile = None
    nowData = None
    nowFile = None
    for ind, obj in enumerate(objectOrder):
        if lastfile == fileOrder[ind]:
            nowData = lastdata
            # nowFile = lastfile
        else:
            for path in filepathlist:
                if os.path.exists(path + fileOrder[ind] + '.hdr'):
                    # enviData = envi.open(path + file + '.hdr', path + file + '.img')
                    data = envi.open(path + fileOrder[ind] + '.hdr', path + fileOrder[ind] + '.img')
                    data = data.load()
                    nowData = np.array(data)
                    lastdata = nowData
                    break
        lastfile = fileOrder[ind]
        line = nowData[posOrder[ind][1], posOrder[ind][0]]
        line = line / np.max(line)
        plt.plot(wavelength, line, label=objectOrder[ind])
    plt.xlabel("波长/nm")
    plt.ylabel("归一化光谱响应强度值")
    # plt.title('各场景下太阳入射光光谱响应强度归一化曲线')
    # legend_font = {
    #     'family': 'Times New Roman',  # 字体
    #     'style': 'normal',
    #     'size': 10,  # 字号
    #     'weight': "normal",  # 是否加粗，不加粗
    # }
    # plt.legend(loc='upper right',frameon = False)  # 打上标签
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)

    plt.savefig('各物体归一化光谱响应强度值对比图.png', bbox_inches='tight')
    plt.show()


    # 获取白板数据
    # for i, file in enumerate(filelist):
    #     data = envi.open(dirpath + file + '.hdr', dirpath + file + '.img')
    #     data = data.load()
    #     data = np.array(data)
    #     data = envi_normalize(data)
    #     print(data.shape)
    #     pos = posList[i]
    #     print(pos)
    #     plt.plot(wavelength, data[pos[1], pos[0]], label=legendList[i])
    #     # plt.legend(legendList[i])
    # plt.xlabel("波长/nm")
    # plt.ylabel("归一化光谱响应强度值")
    # plt.title('各场景下太阳入射光光谱响应强度归一化曲线')
    # # plt.legend(legendList)  # 打上标签
    # plt.legend()
    # plt.savefig('各场景归一化2.png')
    # plt.show()

    # qingtianData = envi.open(dirpath + yintianyingyinImg + '.hdr', dirpath + yintianyingyinImg + '.img')
    # print(qingtianData.shape)
    # plt.plot(wavelength, qingtianData[yintianyingyinPos[::-1]])
    # plt.xlabel("波长/nm")
    # plt.ylabel("光谱响应强度值")
    # plt.title('阴天阴影处太阳入射光光谱曲线')
    # plt.legend()  # 打上标签
    # plt.savefig('阴天阴影.png')
    # plt.show()

