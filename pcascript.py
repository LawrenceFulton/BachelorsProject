from sklearn import decomposition as decomp
from sklearn.cluster import KMeans
import numpy as np
import random
from matplotlib import pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

n_indi = 1000
n_desicion = 1000

data = np.zeros((n_indi,n_desicion))

for r in range(n_indi):
    for c in range(n_desicion):
        data[r,c] = random.choice([-1,0,1])




out = decomp.PCA(n_components=2).fit_transform(data)

kmenas = KMeans(n_clusters=2).fit(out)



print(out)
fig, ax = plt.subplots()
for g in np.unique(kmenas.labels_):
    ix = np.where(kmenas.labels_ == g)
    ax.scatter(out[ix, 0], out[ix, 1], s=1, label=g)
plt.savefig("pca")




fig = plt.figure(1, figsize=(16, 9))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = decomp.PCA(n_components=3).fit_transform(data)
for g in np.unique(kmenas.labels_):
    ix = np.where(kmenas.labels_ == g)
    ax.scatter(X_reduced[ix, 0], X_reduced[ix, 1], X_reduced[ix, 2], edgecolor='k', s=40, label=g)

plt.savefig("pca3.pdf")