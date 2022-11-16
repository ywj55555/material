import os


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


if __name__ == "__main__":
    mkdir('./test_mkdir/')
    mkdir('test_mkdir\\')
