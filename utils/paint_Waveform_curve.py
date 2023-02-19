import spectral.io.envi as envi
import os
import numpy as np
import gc
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import (manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection)

# 皮肤重新找几张！
# skin_file = ['20210329145902788','20210329145944767','20210329145952293']
# skin_coor = [[[638,850],[659,856],[1443,744],[869,981]],[[1060,695],[1072,751]],[[1004,590],[1067,1154],[1096,1115],[576,666]]]

skin_file = ['20210521101100292','20210521101147836','20210521095839946']
skin_coor = [[[687, 1098], [455, 1141], [687, 970], [468, 1217]],[[782, 333], [790, 273], [815, 326]],[[843, 550], [877, 525],[831, 557]]]

sky_file = ['20210329150824296','20210521131824332','20210329150659428']
sky_coor = [[[1163,469],[26,16],[868,434],[202,186],[332,503]],[[152,189],[393,105]],[[32,312],[291,27],[146,725]]]

plant_file = ['20210521095803026','20210521100215003','20210521102605642']
plant_coor = [[[1152,827],[1577,1003],[303,954],[417,210]],[[166,496],[44,672]],[[822,803],[477,670],[863,359],[29,115]]]
# plant_coor = [[[1152,827],[1196,950],[1577,1003],[303,954],[518,984],[417,210]],[[166,496],[44,672]],[[822,803],[477,670],[863,359],[29,115]]]

cloth_file = ['20210521095839946','20210521102600411','20210521102605642']
cloth_coor = [[[867,432],[706,660],[667,495],[708,367],[736,515]],[[1044,921],[964,999]],[[1038,485],[1255,523],[1341,504]]]
# cloth_coor = [[[867,432],[911,446],[855,638],[706,660],[667,495],[708,367],[869,368],[855,370],[715,367],[736,515]],[[1044,921],[1063,893],[964,999],[963,982]],[[1038,485],[1255,523],[1341,504]]]

water_file = ['20220427144846394', '20220604125805482', '20220429172605504']
water_coor = [[[1085, 899], [544, 862], [665, 510], [427, 1210]], [[335, 1126], [876, 1335], [1267, 1296]], [[532, 1040], [1469, 1119], [645, 619]]]

dirpath_sh = r'F:\shanghai-multispectral\envi'+'\\'
dirpath_hz = r'F:\hangzhou\envi'+'\\'
dirpath_water1 = r'E:\shenzhen\img\20220427'+'\\'
dirpath_water2 = r'E:\shenzhen\img\20220429'+'\\'
dirpath_water3 = r'E:\hefei\img'+'\\'
filepathlist = [dirpath_sh, dirpath_hz, dirpath_water1, dirpath_water2, dirpath_water3]
# dirpath = '/home/cjl/data/envi/'

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


def envi_normalize(imgData):
    img_max = np.max(imgData, axis=1,keepdims=True) #通道归一化
    # return imgData / img_max[:, :, np.newaxis]
    return imgData / img_max

def plot_embedding(X, y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    # ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]), color=plt.cm.Set1(y[i] / 10.), fontdict={'weight': 'bold', 'size': 9})

if __name__ == '__main__':
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei']
    namelist = ['skin', 'cloth', 'water']
    # data = []
    file_list = []
    coor_list = []
    file_list.append(skin_file)
    # file_list.append(sky_file)
    # file_list.append(plant_file)
    file_list.append(cloth_file)
    file_list.append(water_file)

    coor_list.append(skin_coor)
    # coor_list.append(sky_coor)
    # coor_list.append(plant_coor)
    coor_list.append(cloth_coor)
    coor_list.append(water_coor)

    for ind,file_co in enumerate(file_list):
        data = []
        for i,file in enumerate(file_co):
            enviData = None
            for path in filepathlist:
                if os.path.exists(path + file + '.hdr'):
                    enviData = envi.open(path + file + '.hdr', path + file + '.img')
                    break
                # else if :
                #     enviData = envi.open(dirpath_hz + file + '.hdr', dirpath_hz + file + '.img')
            # enviData = envi.open(dirpath + file + '.hdr', dirpath + file + '.img')
            imgData = enviData.load()
            imgData = np.array(imgData,dtype=np.float64)
            print(imgData.shape)
            for coor in coor_list[ind][i]:
                data.append(imgData[coor[1],coor[0]])

            gc.collect()
        # x = [i for i in range(1,129)]

        for line in data:
            plt.plot(wavelength, line)
        plt.xlabel("波长/nm")
        plt.ylabel("光谱响应强度值")
        plt.savefig('./res/' + namelist[ind] + '.png')
        plt.clf()

        for line in data:
            line = line / np.max(line)
            plt.plot(wavelength, line)
        plt.xlabel("波长/nm")
        plt.ylabel("归一化光谱响应强度值")
        plt.savefig('./res/' + namelist[ind] + '_nor.png')
        plt.clf()
        # data = np.array(data)
    # print(data.shape)
    # labels = []
    # labels.extend([x for x in range(1,5) for i in range(10) ])
    # np.savez("data_label.npz", data=data, labels=labels)
    # tsne = TSNE(n_components=2, random_state=0)
    # Y = tsne.fit_transform(data)
    # plt.scatter(Y[:, 0], Y[:, 1], s=2, c=labels)
    # plt.savefig('ori_scatter_random.png')
    # # plt.savefig(str(n_components) + '-' + source + '.png')
    #
    # plt.figure()
    # data_nor = envi_normalize(data)
    # data_nor = data_nor*5000
    # tsne = TSNE(n_components=2, random_state=0)
    # Y = tsne.fit_transform(data_nor)
    # plt.scatter(Y[:, 0], Y[:, 1], s=2, c=labels)
    # plt.savefig('nor_scatter_random.png')
    # print(data.shape)
    # p1 = plt.subplot(121)
    # fig = plt.figure(0)
    # for
    # for k in range(10):
    #     plt.plot(wavelength,skin_data[k])
    # # plt.plot(x, y1, label="sin")
    # # plt.plot(x, y2, label="cos", linestyle="--")
    # plt.xlabel("波段序号")
    # plt.ylabel("像素强度值")
    # # p2 = plt.subplot(122)
    # plt.savefig('sky.png')
    # plt.close()
    # skin_data = envi_normalize(skin_data)
    # # fig = plt.figure(1)
    # for k in range(10):
    #     plt.plot(x,skin_data[k])
    # # plt.xlabel("波段序号")
    # plt.ylabel("归一化强度值")

    # plt.title('sin & cos')
    # plt.legend()  # 打上标签
    # plt.savefig('sky_nora.png')
    # plt.show()
    # print(skin_data.shape)
    # print(np.max(skin_data),np.min(skin_data))



# print(len(skin_coor))
# for i in range(len(skin_coor)):
#     print(len(skin_coor[i]))