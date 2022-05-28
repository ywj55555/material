import os

class HdrDataTypeError(Exception):
    def __init__(self,error_info):
        self.error_info = error_info
    def __str__(self):
        print("HdrDataTypeError : ", self.error_info)

def judegHdrDataType(hdr_dirpath, file):
    with open(hdr_dirpath +"/"+ file + '.hdr', "r") as f:
        data = f.readlines()
    modify_flag = False
    if data[5] != 'data type = 12\n':
        data[5] = 'data type = 12\n'
        modify_flag = True
        # raise HdrDataTypeError("data type = 2, but data type should be 12")
    if modify_flag:
        with open(hdr_dirpath + "/" + file + '.hdr', "w") as f:
            f.writelines(data)
        print("mend the datatype of file : ", file)

if __name__ == '__main__':
    path = 'D:/dataset/lgimg/train/'
    file = '20220429173118989'
    judegHdrDataType(path, file)
    # print("successful pass")


