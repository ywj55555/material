import spectral.io.envi as envi
import os
import numpy as np
import gc
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import (manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection)

skin_file = ['20210329145902788','20210329145944767','20210329145952293']
skin_coor = [[[638,850],[659,856],[1443,744],[869,981]],[[1060,695],[1072,751]],[[1004,590],[1067,1154],[1096,1115],[576,666]]]

sky_file = ['20210329150824296','20210521131824332','20210329150659428']
sky_coor = [[[1163,469],[26,16],[868,434],[202,186],[332,503]],[[152,189],[393,105]],[[32,312],[291,27],[146,725]]]

plant_file = ['20210521095803026','20210521100215003','20210521102605642']
plant_coor = [[[1152,827],[1577,1003],[303,954],[417,210]],[[166,496],[44,672]],[[822,803],[477,670],[863,359],[29,115]]]
# plant_coor = [[[1152,827],[1196,950],[1577,1003],[303,954],[518,984],[417,210]],[[166,496],[44,672]],[[822,803],[477,670],[863,359],[29,115]]]

cloth_file = ['20210521095839946','20210521102600411','20210521102605642']
cloth_coor = [[[867,432],[706,660],[667,495],[708,367],[736,515]],[[1044,921],[964,999]],[[1038,485],[1255,523],[1341,504]]]
# cloth_coor = [[[867,432],[911,446],[855,638],[706,660],[667,495],[708,367],[869,368],[855,370],[715,367],[736,515]],[[1044,921],[1063,893],[964,999],[963,982]],[[1038,485],[1255,523],[1341,504]]]

# dirpath_sh = r'E:\shanghai-multispectral\envi'+'\\'
# dirpath_hz = r'E:\hangzhou\envi'+'\\'
dirpath = '/home/cjl/data/envi/'

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
    data = []
    file_list = []
    coor_list = []
    file_list.append(skin_file)
    file_list.append(sky_file)
    file_list.append(plant_file)
    file_list.append(cloth_file)

    coor_list.append(skin_coor)
    coor_list.append(sky_coor)
    coor_list.append(plant_coor)
    coor_list.append(cloth_coor)


    for ind,file_co in enumerate(file_list):
        for i,file in enumerate(file_co):
            # if os.path.exists(dirpath_sh + file + '.hdr'):
            #     enviData = envi.open(dirpath_sh + file + '.hdr', dirpath_sh + file + '.img')
            # else:
            #     enviData = envi.open(dirpath_hz + file + '.hdr', dirpath_hz + file + '.img')
            enviData = envi.open(dirpath + file + '.hdr', dirpath + file + '.img')
            imgData = enviData.load()
            imgData = np.array(imgData,dtype=np.float64)
            print(imgData.shape)
            for coor in coor_list[ind][i]:
                data.append(imgData[coor[1],coor[0]])
            gc.collect()
        # x = [i for i in range(1,129)]
    data = np.array(data)
    labels = []
    labels.extend([x for x in range(1,5) for i in range(10) ])
    np.savez("data_label.npz", data=data, labels=labels)
    tsne = TSNE(n_components=2, random_state=0)
    Y = tsne.fit_transform(data)
    plt.scatter(Y[:, 0], Y[:, 1], s=2, c=labels)
    plt.savefig('ori_scatter_random.png')
    # plt.savefig(str(n_components) + '-' + source + '.png')

    plt.figure()
    data_nor = envi_normalize(data)
    data_nor = data_nor*5000
    tsne = TSNE(n_components=2, random_state=0)
    Y = tsne.fit_transform(data_nor)
    plt.scatter(Y[:, 0], Y[:, 1], s=2, c=labels)
    plt.savefig('nor_scatter_random.png')
    print(data.shape)
    # p1 = plt.subplot(121)
    # fig = plt.figure(0)
    # for k in range(10):
    #     plt.plot(x,skin_data[k])
    # plt.plot(x, y1, label="sin")
    # plt.plot(x, y2, label="cos", linestyle="--")
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