import os
import shutil

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

def judegHdrDataType(hdr_dirpath, file):
    with open(hdr_dirpath +"/"+ file + '.hdr', "r") as f:
        data = f.readlines()
    modify_flag = False
    if data[5] != 'data type = 12\n':
        data[5] = 'data type = 12\n'
        modify_flag = True
        # raise HdrDataTypeError("data type = 2, but data type should be 12")

    if data[6].split(' =')[0] != 'byte order':
        data.insert(6, 'byte order = 0\n')
        modify_flag = True
    else:
        if data[6] != 'byte order = 0\n':
            data[6] = 'byte order = 0\n'
            modify_flag = True
    if modify_flag:
        with open(hdr_dirpath + "/" + file + '.hdr', "w") as f:
            f.writelines(data)
        print("mend the datatype of file : ", file)

if __name__ == "__main__":
    # mkdir('./test_mkdir/')
    # mkdir('test_mkdir\\')
    dirpath_water1 = r'E:\shenzhen\img\20220427' + '\\'
    dirpath_water2 = r'E:\shenzhen\img\20220429' + '\\'
    dirpath_water3 = r'E:\hefei\img' + '\\'
    dstpath = r'F:\spectralDateset\water'+ '\\'
    water_file = ['20220427144846394', '20220604125805482', '20220429172605504']
    filepathlist = [ dirpath_water1, dirpath_water2, dirpath_water3]
    for file in water_file:
        for path in filepathlist:
            if os.path.exists(path + file + '.hdr'):
                shutil.copy(path + file + '.hdr', dstpath + file + '.hdr')
                shutil.copy(path + file + '.img', dstpath + file + '.img')
                # enviData = envi.open(path + file + '.hdr', path + file + '.img')
                break


