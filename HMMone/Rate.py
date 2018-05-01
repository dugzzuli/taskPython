from sklearn import metrics
import numpy as np

labels_true = np.loadtxt('./truelabel.txt').T
labels_pred = np.loadtxt('./label.txt').T
print('labels_true:')
print(np.shape(labels_true))
mu=metrics.adjusted_mutual_info_score(labels_true, labels_pred)
print(mu)
ri=metrics.adjusted_rand_score(labels_true, labels_pred)
print(ri)