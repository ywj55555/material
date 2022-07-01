import os


def auto_modify_hdr(hdr_dirpath):
    for dirpath, dirnames, filenames in os.walk(hdr_dirpath, topdown=False):
        for file in filenames:
            if file[-4:] != ".hdr":
                continue
            print(file)
            with open(hdr_dirpath + file, "r") as f:
                data = f.readlines()
                f.close()
            modify_flag = False
            for ind in range(len(data)):
                if data[ind] == 'data type = 2\n':
                    data[ind] = 'data type = 12\n'
                    modify_flag = True
            if modify_flag:
                with open(hdr_dirpath + file, "w") as f:
                    f.writelines(data)
                    f.close()

if __name__ == '__main__':
    hdr_dirpath = 'E:/qiang/envi/'
    auto_modify_hdr(hdr_dirpath)