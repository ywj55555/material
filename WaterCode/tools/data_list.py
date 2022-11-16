import os

dataset_dir = '/home/glk/datasets/hefei/All_data/label/'
img_list = []
for filename in os.listdir(dataset_dir):
    if  filename[-4:] == '.png':#filename[-4:] == '.img':# or filename[-4:] == '.tif'
        img_list.append(filename[:-4])#'rgb' +
        # print(filename[:-4])
# with open(dataset_dir + "test.txt","w") as f:
#     for i in img_list:
#         f.write(i+'\n')

dataset_dir_A = '/home/glk/datasets/hefei/All_data/'
img_list_A = []
for filename in os.listdir(dataset_dir_A):
    if  filename[-4:] == '.img':#filename[-4:] == '.png':# or filename[-4:] == '.tif'
        img_list_A.append('rgb' +filename[:-4])#

for i in img_list:
    if i in img_list_A:
        pass
        # continue
    else:
        print(i)