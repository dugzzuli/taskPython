# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        print(root) #当前目录路径
        print(dirs) #当前路径下所有子目录
        print(files) #当前路径下所有非目录子文件
        return files
def dir_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        print(root) #当前目录路径
        print(dirs) #当前路径下所有子目录
        print(files) #当前路径下所有非目录子文件
        return dirs
def loadCSVfile(path):
    data_csv = pd.read_table(path,sep='\t', header=None)
    print(data_csv.values)

    return data_csv
def loadCSVfile2(path):
    tmp = np.loadtxt(path, dtype=np.float32, delimiter="\t")
    print(tmp)
    data = tmp[:,:].astype(np.int32)#加载数据部分

    return data
if __name__ == '__main__':
    Allpath="../data/"
    dirs = dir_name("../data/")
    for i in range(len(dirs)):
        print(Allpath+dirs[i])
        files=file_name(os.path.join(Allpath,dirs[i]))
        print(files)
        length=len(files);
        d0=np.zeros([length,15])
        d1=np.zeros([length,15])
        d2=np.zeros([length,15])
        d3=np.zeros([length,15])
        d4=np.zeros([length,15])
        d5=np.zeros([length,15])

        dataAll=[]
        for file in range(len(files)):
            path=os.path.join(os.path.join(Allpath,dirs[i]), files[file])
            data=loadCSVfile2(path)
            dataT=data.T

            path = os.path.join("../ReData/"+dirs[i], files[file])
            np.savetxt(path,dataT,fmt="%d", delimiter='\t')
            [m,n]=np.shape(dataT)
            d0[file,:]=dataT[0,:]
            d1[file,:] = dataT[1, :]
            d2[file,:] = dataT[2, :]
            d3[file,:] = dataT[3, :]
            d4[file,:] = dataT[4, :]
            d5[file,:] = dataT[5, :]
        dataAll=[d0,d1,d2,d3,d4,d5]
