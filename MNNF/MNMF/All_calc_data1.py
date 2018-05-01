from numpy import *
import numpy as np
import operator
from dtw import dtw
from sklearn import preprocessing
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
import os
import numpy as np
import pandas as pd

def dir_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        print(root) #当前目录路径
        print(dirs) #当前路径下所有子目录
        print(files) #当前路径下所有非目录子文件
        return dirs
def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        print(root) #当前目录路径
        print(dirs) #当前路径下所有子目录
        print(files) #当前路径下所有非目录子文件
        return files
def loadCSVfile3(path,flag="\t"):
    data_csv = pd.read_table(path,sep=flag, header=None)
    
    return data_csv
def loadCSVfile2(path,flag="\t"):
    tmp = np.loadtxt(path, dtype=np.str, delimiter=flag)
    data = tmp[:,:].astype(np.float)#加载数据部分

    return data
def loadCSVfile(path,flag="\t"):
    tmp = np.loadtxt(path, dtype=np.str, delimiter=flag)
    data = tmp[1:,:-1].astype(np.float)#加载数据部分
    label = tmp[1:,-1].astype(np.float)#加载类别标签部分
    # data = tmp[:,:].astype(np.float)#加载数据部分
    # np.savetxt("data.txt",data,delimiter='\t')
    return data, label #返回array类型的数据
def my_custom_norm(x, y):
    return (x * x) + (y * y)
def my_custom_norm_3(x, y):
    return (x * x) + (y * y)
def my_custom_norm_edu(x, y):
    return np.sqrt((x * x) + (y * y))
def norm_L1(x, y):
    return np.abs(x-y)
def calDistance(x,y,distance):

    dist, cost, acc, path = dtw(x, y, dist=distance)
#    dist, cost, acc, path = dtwdemo(x, y, dist=my_custom_norm)
    return dist
def cal_data_scale(data,distance=my_custom_norm_edu):

    m,n=np.shape(data)
    distanceMat = np.zeros([m, m])
    fit=preprocessing.StandardScaler();
    distanceMat_stand=fit.fit_transform(data)
    for i in range(m):
        for j in range(m):
            if (i!=j):
                distanceMat[i,j]=calDistance(distanceMat_stand[i,:],distanceMat_stand[j,:],distance);
    return distanceMat

def cal_data(data,distance=my_custom_norm_edu):

    m,n=np.shape(data)
    distanceMat = np.zeros([m, m])

    for i in range(m):
        for j in range(m):
            if (i!=j):
                distanceMat[i,j]=calDistance(data[i,:],data[j,:],distance);
    return distanceMat
def norm_distance(distance):
    L=np.tril(distance,-1)
    min_max_scaler = preprocessing.MinMaxScaler()
    distance_tran=min_max_scaler.fit_transform(L)
    distance_tran=distance_tran+np.transpose(distance_tran)
    return distance_tran
def net_creat_dist(data,epsilon,flag=1):
    datamat=np.zeros(np.shape(data))
    datamat[data <epsilon] = 1;
    [m,n]=np.shape(datamat)
    if flag==1:
        for i in range(m):
            for j in range(n):
                if(i==j):
                    datamat[i,j]=0;
    return datamat
def net_creat_dist_K(data,epsilon,flag=1):
    datamat=np.zeros(np.shape(data))
    [m, n] = np.shape(datamat)
    ndarray.sort(axis=-1, kind='quicksort', order=None)
    for i  in range(m):
        A=np.argsort(m[i,:])


    return datamat
def draw_net(net,i):
    graph = nx.from_numpy_matrix(net)
    # nx.draw(graph, pos=nx.random_layout(graph), node_color='b', edge_color='r', with_labels=True, font_size=18,
    #         node_size=20)
    nx.write_pajek(graph,i+'Pajek.net')
    # plt.savefig("networkpic/wangluotu1.jpg")

def kmeans_Demo(net,n_clusters=2):
    clf = KMeans(n_clusters=n_clusters)
    s = clf.fit(net)
    print(s)
    # 9个中心
    print(clf.cluster_centers_)
    # 每个样本所属的簇
    print(clf.labels_)
    # 用来评估簇的个数是否合适，距离越小说明簇分的越好，选取临界点的簇个数
    print("用来评估簇的个数是否合适，距离越小说明簇分的越好，选取临界点的簇个数")
    print(clf.inertia_)

def mkdir(path):
    # 引入模块
    import os

    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)

        print
        path + ' 创建成功'
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print
        path + ' 目录已存在'
        return False
if __name__ == "__main__":
    Allpath="./ReData/"
    Save_path="./Save/"
    dirs=dir_name(Allpath)
    # for dir_name in dirs:
    #     current_dir=os.path.join(Allpath,dirs)
    #     Save_path_dir=os.path.join(Save_path,dirs)
    #     files = file_name(current_dir)
    #     length = len(files);
    #     d0 = np.zeros([length, 15])
    #     d1 = np.zeros([length, 15])
    #     d2 = np.zeros([length, 15])
    #     d3 = np.zeros([length, 15])
    #     d4 = np.zeros([length, 15])
    #     d5 = np.zeros([length, 15])
    #     dataAll = []
    #     label=zeros([length,1])
    #     for file in range(len(files)):
    #         path = os.path.join(current_dir, files[file])
    #         if(path.__contains__('c')):
            #     label[file,0]=1
            # elif (path.__contains__('n')):
            #     label[file, 0] = 2
            # elif (path.__contains__('o')):
            #     label[file, 0] = 3
    #         dataT = loadCSVfile2(path)
    #         [m, n] = np.shape(dataT)
    #         d0[file, :] = dataT[0, :]
    #         d1[file, :] = dataT[1, :]
    #         d2[file, :] = dataT[2, :]
    #         d3[file, :] = dataT[3, :]
    #         d4[file, :] = dataT[4, :]
    #         d5[file, :] = dataT[5, :]
    #     dataAll = [d0, d1, d2, d3, d4, d5]
        # 将组装之后的数据文件进行保存
        # for i in range(len(dataAll)):
        #     zuzhuang_path=os.path.join(Save_path_dir,'dataAll')
        #     mkdir(zuzhuang_path)
        #     np.savetxt(zuzhuang_path+"data" + str(i) + ".txt", dataAll[i],fmt='%f', delimiter='\t')
        # step=0.001
        # for p in range(1):
        #     for i in range(len(dataAll)):
        #         print("正在生成距离文件中。。。。。。" + str(i))
        #         d = cal_data(dataAll[i],my_custom_norm)
        #         distance_path=os.path.join(Save_path_dir,"/cal_data/"+str(p+1)+"/")
        #         mkdir(distance_path);
        #         np.savetxt(distance_path+str(p+1)+"/data" + str(i) + ".txt", d, fmt='%f', delimiter='\t')
        #         print("正在生成标准化距离文件中。。。。。。" + str(i))
        #         distance_norm = norm_distance(d)
        #         distance_norm_path=os.path.join(Save_path_dir,"/norm_cal_data/"+str(p+1)+"/")
        #         mkdir(distance_norm_path);
        #         np.savetxt(distance_norm_path+str(p+1)+"/data" + str(i) + ".txt", distance_norm, fmt='%f', delimiter='\t')
        #         for enn in range(1000):
        #             print("正在生成eNN网络文件中。。。。。。" + str(i))
        #             net = net_creat_dist(distance_norm, (enn+1)*step,flag=1);
        #             calc_net=os.path.join(Save_path_dir,"/calc_net/" + str(enn + 1) + "/")
        #             mkdir(calc_net);
        #             np.savetxt(calc_net+"/data" + str(i) + ".txt", net, fmt='%f', delimiter='\t')
        # break
