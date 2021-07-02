import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering as Agg
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA



def ideal_n_cluster(data, alg = "k"):
    n_paticipants = data.shape[0]

    sil_score = []
    y = []
    best_label = None
    best_n_cluster = -1
    best_score = -1

    for i in range(2, min(20, n_paticipants)):

        if alg == "k":
            labels = k_clustering(data, i)
        else:
            labels = agg_clustering(data, i)
        
        score = silhouette_score(data, labels)
        
        if score > best_score:
            best_score = score
            best_n_cluster = i
            best_label = labels

        sil_score.append(score)
        y.append(i)

    plt.plot( y, sil_score)
    plt.savefig("tmp/sil_score")
    plt.close()

    print("BEST: ", best_n_cluster, best_label)
    print(sil_score)

    return best_label, best_n_cluster

def k_clustering(data, n_clusters = 2):
    labels = KMeans(n_clusters=n_clusters).fit(data).labels_
    labels = np.array(labels)
    return labels


def agg_clustering(data, n_clusters = 2):
    labels = Agg(n_clusters=n_clusters).fit(data).labels_
    labels = np.array(labels)
    return labels


def dimen_reduc(data, dimen = 2):
    out = PCA(dimen).fit_transform(data)
    return out
