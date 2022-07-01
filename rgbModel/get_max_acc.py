import os
filepath = r'./output_300m2.log'
best_test_acc = 0
# best_test_epoch = 0

with open(filepath,'r')as f:
    lines = f.readlines()
    for line in lines:
        if line.find('all test accuracy')!=-1:
            # print(line)
            # sf.write(line+'\n')
            # if line.find('test') != -1:
            tmp = float(line[line.find(':') + 2:].split(" ")[0])
            if best_test_acc < tmp:
                best_test_acc = tmp
                # best_test_epoch = line[line.find('epoch') + 7:]

print(best_test_acc)