import numpy as np
import random as rd
from sklearn import decomposition as decomp
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from scipy.spatial import distance

rd.seed(0)

n_desicion = 100
n_indi = 100
# proto1 = np.random.rand(n_desicion)
# proto1 = (proto1 - 0.5)*2

# proto2 = np.random.rand(n_desicion)
# proto2 = (proto2 - 0.5)*2

# proto2 = proto1 *  -1

proto1 = np.array([1,1,-1,1])
proto2 = np.array([-1,0,-1,0])



data = np.zeros((n_indi,n_desicion))
belives_arr = np.zeros(n_indi)


for r in range(n_indi):
    belive = rd.random()
    # belive = r/10
    belives_arr[r] = belive
    data[r,:] = (belive * proto1 + (1-belive) * proto2)  + (rd.random()- 0.5)


print(proto1)
print(proto2)
print(proto1-proto2)


trans_data = data.transpose()

std_dev = np.std(data,axis=0)
print(std_dev)

dis_mat = distance.cdist(trans_data, trans_data)
print(dis_mat)




# data = np.vstack((data, proto1, proto2))

# print(data)
# data = np.where(data < 0.33, data, 1 )
# data = np.where(data > -0.33, data, -1 )
# data = np.round(data)
# print(data)



# pca = decomp.PCA(n_components=2)
# pca.fit(data)
# out = pca.transform(data)


# kmenas = KMeans(n_clusters=2).fit(out)

# kmenas.labels_[n_indi] = 99
# kmenas.labels_[n_indi+1] = 100

# fig, ax = plt.subplots()
# for g in np.unique(kmenas.labels_):
#     ix = np.where(kmenas.labels_ == g)
#     if (g == 99 or g == 100):
#         ax.scatter(out[ix, 0], out[ix, 1], s=10, label=g)
#     else:
#         ax.scatter(out[ix, 0], out[ix, 1], s=1, label=g)


# plt.legend()
# plt.savefig("pca.pdf")
