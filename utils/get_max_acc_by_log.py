file = '../output_twoBranch3.0_7_SkinClothWater18_AllSelectedExtraTest4.log'
save_txt = './output_get_maxSkinRec_SkinClothWater.txt'
# sf = open(save_txt,'w')
best_test_acc = 0.0
best_test_skin_rec = 0.0
best_test_skin_f1 = 0.0
best_test_epoch = ''
epoch_acc_dict = {}
with open(file,'r')as f:
    lines = f.readlines()
    # print(len(lines),'  jj')
    for line in lines:
        # if line.find('skin_') != -1 or line.find('rate')!=-1:
            # print(line)
            # sf.write(line+'\n')
        if line.find('micro accuracy') != -1:
            pos = line.find('accuracy')
            res = float(line[pos:].split(" ")[1])
            if res > best_test_acc:
                # print(line[line.find(':') + 3:]
                best_test_acc = res
        if line.find('cloth') != -1:
            # if best_test_acc < float(line[line.find('epoch') + 14:].split(" ")[0]):
            #     best_test_acc = float(line[line.find('epoch') + 14:].split(" ")[0])
            #     best_test_epoch = line[line.find('epoch') + 7:]
            # beginpos = line.find(':')
            # epoch = line[line.find('epoch') + 7:].split(" ")[0]
            # acc = float(line[line.find(':', beginpos + 1) + 3:])
            # epoch_acc_dict[epoch] = acc
            rec_pos = line.find('rec')
            res = float(line[rec_pos:].split(" ")[1])
            f1_pos = line.find('f1-score')
            f1_score = float(line[f1_pos:].split(" ")[1])
            if res > best_test_skin_rec:
                # print(line[line.find(':') + 3:]
                best_test_skin_rec = res
            if f1_score > best_test_skin_f1 and f1_score > 0.79 and f1_score < 0.81:
                best_test_skin_f1 = f1_score
                # best_test_acc = float(line[line.find(':', beginpos + 1) + 3:])
            #     best_test_epoch = line[line.find('epoch') + 7:].split(" ")[0]
print(best_test_skin_rec)
print(best_test_skin_f1)
print(best_test_acc)
# epoch_acc_dict = sorted(epoch_acc_dict.items(), key=lambda item: item[1], reverse=True)
# print(epoch_acc_dict[:20])
# for key in epoch_acc_dict[:20]:
#     print(key[0], end=", ")
# lines = sf.readlines()
# best_test_acc = 0.0
# best_test_epoch = ''
# for line in lines:
#     if line.find('test')!=-1:
#         if best_test_acc<float(line[line.find('acc')+7:].strip()):
#             best_test_acc =float(line[line.find('acc')+7:].strip())
#             best_test_epoch = line[line.find('epoch')+7:]
# print(best_test_acc)
# print(best_test_epoch)