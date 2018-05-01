import numpy as np
from sklearn.cluster import KMeans

from HMMone import KLD

mat=np.loadtxt("mat.txt")

m,n=mat.shape
H=np.zeros([m,n])
print(m,n)
# min_max_scaler = preprocessing.MinMaxScaler()
# H = min_max_scaler.fit_transform(mat)
# print(H)

for i in range(m):
    for j in range(n):
        H[i,j]=mat[i,j]/sum(mat[:,j])
print(H)
a=sum(H)
print(a)
print(sum(sum(H)))
KLMat=np.zeros([m,n])
for i in range(m):
    for j in range(n):
        KLMat[i,j]= KLD.symmetricalKL(H[:, i], H[:, j])

print(KLMat)

kmeans = KMeans(n_clusters=3, random_state=0).fit(KLMat)
np.savetxt("./label.txt",kmeans.labels_)




