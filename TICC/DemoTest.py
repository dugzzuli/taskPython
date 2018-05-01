import numpy as np
data=np.loadtxt("./example_data.txt",delimiter=',')
print(data.shape)
cluster=np.loadtxt("./Results.txt",delimiter=',')
print(cluster.shape)
print(np.unique(cluster))
