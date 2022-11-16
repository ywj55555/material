from skimage import io


code2label = [0,3,3,3,1,3,3,3,2,1,1,1,1,1,1,4,4,4,1,1,1,1,1,1,1,1,1,1,1,1,1] #2:skin 3:cloth 4:plant 1:other 0：no-train
label_path = '/home/glk/datasets/Multispec/label/'
label_save_path = '/home/glk/datasets/Multispec/label_4/'
data_file = '/home/glk/datasets/Multispec/' + 'train_all.txt'

with open(data_file, 'r') as f:
    dataFile = f.readlines()

img_list = dataFile
length = len(img_list)

for i in range(length):
    label = io.imread(label_path + img_list[i].split('/')[-1].split('\n')[0] + '.png')
    H,W = label.shape
    for h in range(H):
        for w in range(W):
            label[h,w] = code2label[label[h,w]]
    # print(label.shape)
    io.imsave(label_save_path + img_list[i].split('/')[-1].split('\n')[0] + '.png',label)
    # label_tmp = io.imread(label_save_path + img_list[i].split('/')[-1].split('\n')[0] + '.png')
    # print(label_tmp.shape)