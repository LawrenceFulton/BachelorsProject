from math import sqrt
import numpy as np
import conversation as con 
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans, AgglomerativeClustering as Agg
from sklearn.metrics import silhouette_score
import clustering as cls



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



def binarising_labels(labels, true_value):
    bin_label = (labels == true_value)
    print(bin_label)
    return bin_label

def plot_clustering(data, labels):

    out = PCA(n_components=2).fit_transform(data)
    print("labels", labels)

    print("PCA", out)

    fig, ax = plt.subplots()
    for g in np.unique(labels):
        ix = np.where(labels == g)
        ax.scatter(out[ ix,0], out[ix,1], s=1, label=g)

    plt.legend()
    plt.savefig("tmp/ideal_n_cluster")
    


def two_prop_test(succ_in, succ_out, pop_in ,pop_out ):
    '''
    function taken from stats.clj
    '''

    pi1 = succ_in / pop_in
    pi2 = succ_out / pop_out
    pi_hat = (succ_in + succ_out) / (pop_in + pop_out)
    if pi_hat == 1:
        return 0 
    z = (pi1 - pi2) / sqrt(pi_hat * (1 - pi_hat) * ((1 / pop_in) + (1 / pop_out)))
    print(z)


def prop_test(succ, n):
    succ += 1
    n += 1
    out = 2 * sqrt(n) * ((succ / n) + (-0.5))
    print(out)


if __name__ == "__main__":
    data = con.data_creation(100,100, 6)
    ideal_lab, ideal_n_cluser = cls.ideal_n_cluster(data)

    plot_clustering(data, ideal_lab)


    lab_0 = binarising_labels(ideal_lab, 0)
    lab_1 = binarising_labels(ideal_lab, 1)



    ragc_1 = prop_pos(data,lab_0, 1)
    ragc_minus1 = prop_pos(data,lab_0, -1)

    
    print(ragc_1)
    print(ragc_minus1)



    ragc_1 = prop_pos(data,lab_1, 1)
    ragc_minus1 = prop_pos(data,lab_1, -1)

    
    print(ragc_1)
    print(ragc_minus1)


    
    # print(data.shape)



    # print(out.labels_)

    


    

