import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
mat=np.loadtxt("mat.txt")
kmeans = KMeans(n_clusters=3, random_state=0).fit(mat)
# sc= SpectralClustering(n_clusters=3, gamma=0.01).fit(mat)
np.savetxt("./label.txt",kmeans.labels_)
