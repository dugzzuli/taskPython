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
    path="alldata"
    fileList=file_name(path)
    dataMatr=[];
    arrList=[];

    for file in fileList:
        print(file)
        allPath=(os.path.join(path,file))
        arr=np.loadtxt(allPath)[:,:].astype(np.float32)

        arrList.append(arr)
        # model = hmm.GaussianHMM(n_components=2, n_iter=1000, tol=0.001)
        model = hmm.GMMHMM(n_components=1, n_iter=1000, tol=0.001)
        model.fit(arr)
        model.score(arr)
        dataMatr.append(model)
    print("文件长度:")
    print(fileList.__len__())
    mat=np.zeros([fileList.__len__(),fileList.__len__()])
    i=0;
    j=0;
    for model in dataMatr:
        print(model)
        j=0
        for arr in arrList:
            print(i,j)
            mat[i,j]=model.score(arr);
            j=j+1;
        i=i+1;
    print("打印矩阵:")
    print(mat)
    np.savetxt("./mat.txt",mat,delimiter='\t')
    hmm.GaussianHMM

