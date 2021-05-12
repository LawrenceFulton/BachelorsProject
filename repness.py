import numpy as np
import conversation as con 
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans



def prop_pos(data, labels, vote):



    in_group = data[labels]
    out_group = data[~labels]

    # in groupe 

    pos_vote = (in_group == vote)
    all_vote = (in_group != 0)

    nagc = np.sum(pos_vote, axis=0)
    ngc =  np.sum(all_vote, axis=0)

    pagc = (1 + nagc) / (2 + ngc)


    # out groupe 

    pos_vote = (out_group == vote)
    all_vote = (out_group != 0)

    nagc_out = np.sum(pos_vote, axis=0)
    ngc_out =  np.sum(all_vote, axis=0)

    pagc_out = (1 + nagc_out) / (2 + ngc_out)

    ragc = pagc / pagc_out

    return ragc


def clustring(data):
    labels = KMeans(n_clusters=2).fit(data).labels_
    labels = np.array(labels, dtype=bool)
    return labels



if __name__ == "__main__":
    data = con.data_creation(10,10)
    # data = np.array([[1,1,1],[1,1,0],[0,0,0],[0,0,1]])
    labels = clustring(data)

    print(data)

    prop_pos(data)

    # print(data.shape)

    # print(out.labels_)

    


    

