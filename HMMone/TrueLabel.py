import pandas as pd
import numpy as np
from hmmlearn import hmm
import os
def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        print(root) #当前目录路径
        print(dirs) #当前路径下所有子目录
        print(files) #当前路径下所有非目录子文件
        return files
if __name__ == "__main__":
    print("程序开始......")
    path = "alldata"
    fileList = file_name(path)
    label = [];

    for file in fileList:
        print(file)
        if(file.startswith('c')):
            label.append(0)
        elif(file.startswith('n')):
            label.append(1)
        elif(file.startswith('o')):
            label.append(2)
    np.savetxt("./truelabel.txt",label,dtype=np.int32)

