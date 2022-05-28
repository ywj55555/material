# from data.dictNew import *
# from data.dictNew_ori import *
# # print(trainFile_hz)
# # print(trainFile2)
# # b = [y for y in (trainFile_hz + trainFile2) if y not in trainFile_hz] #两个列表中的不同元素
# #
# # # print('a的值为:',a)
# # print('b的值为:',b)
# #
# # c = [x for x in trainFile_hz if x not in trainFile2] #在list1列表中而不在list2列表中
# # d = [y for y in trainFile_hz if y not in trainFile2] #在list2列表中而不在list1列表中
# # print('c的值为:',c)
# # print('d的值为:',d)
# if trainFile_hz==trainFile2:
#     print(True)
# else:
#     print(False)
# if testFile_hz==testFile2:
#     print(True)
# else:
#     print(False)
import numpy as np
seed=2021
np.random.seed(2021)
print(np.random.randint(1,5,(5,5)))