from numpy.core.fromnumeric import shape
from numpy.random import randint
from conversation import data_creation
from repness import prop_pos
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import cluster
import continousConsensus as cc
from sklearn.metrics import silhouette_score
import repness as rep
from sklearn.decomposition import PCA






def cluster_analysis(underlying_data, id): 

    # the max nr of cmts and pers 
    max_cmt = max(underlying_data[:,0]) +1
    max_per = max(underlying_data[:,1]) +1 

    ## data[cmt,per]
    data = np.zeros([max_cmt , max_per])


    bool_cmt = np.zeros(max_cmt, dtype="bool")
    bool_per = np.zeros(max_per, dtype="bool")
    
    scores_2 = []
    scores_3 = []
    scores_4 = []
    idx = []
    cnt = 0
    inc = 10
    goal = 50

    print("length ", underlying_data.shape[0])

    for row in underlying_data:
        cnt += 1 
        cmt_id = row[0]
        per_id = row[1]
        vote = row[2]

        #ubdate of bool arrays 
        bool_cmt[cmt_id] = True
        bool_per[per_id] = True


        #update the known data matrix 
        data[cmt_id,per_id] = vote

        # d1 = data[bool_cmt,:]
        # d2 = d1[:,bool_per]

        
        # print(bool_cmt)

        if (cnt == goal):


            d1 = data[bool_cmt,:]
            d2 = d1[:,bool_per]        
            goal += inc
            inc += 1
            
            idx.append(cnt)
            labels_2 = rep.clustring(d2)
            score_2 = silhouette_score(d2, labels_2)
            scores_2.append(score_2)
            # print(score_2)

            labels_3 = rep.clustring(d2,3)
            score_3 = silhouette_score(d2, labels_3)
            scores_3.append(score_3)
            
            # labels_4 = rep.clustring(d2,4)
            # score_4 = silhouette_score(d2, labels_4)
            # scores_4.append(score_4)

        if cnt % 1000 == 0:
            # update for the cmd 
            print("index", cnt, " and inc ", inc)
        

        if cnt > 150000:


            break



    red_data = PCA(n_components=2).fit_transform(d2)
    # labels_2 = rep.clustring(red_data)
    # print("1:", sum(labels_2)," 2:" , len(labels_2)-sum(labels_2))
    # s1 = silhouette_score(d2, labels_2)    
    # s2 = silhouette_score(red_data, labels_2)    
    # print("prepca:", s1, " postpca:",s2)


    rep.ideal_n_cluster(red_data)
    # print("NICE AMOUNT OF CLUSTER: ", best_n)



    fig, ax = plt.subplots()
    for g in np.unique(labels_2):
        ix = np.where(labels_2 == g)
        # if (g == 99 or g == 100):
        #     ax.scatter(out[ix, 0], out[ix, 1], s=10, label=g)
        # else:
        ax.scatter(red_data[ix, 0], red_data[ix, 1], s=1, label=g)

    plt.legend()
    plt.savefig("tmp/pca")
    plt.close()

   
    print("1:", sum(labels_2)," 2:" , len(labels_2)-sum(labels_2))

    plt.plot(idx,scores_2,  label=id + "2")
    plt.plot(idx, scores_3, label=id + "3")
    # plt.plot(scores_4, label='4')
    plt.legend()

    

    pass







if __name__ == '__main__':
    data, path = cc.preprossessing()

    cluster_analysis(data,"cleaned_data")
    plt.savefig("tmp/sil_" + path+ ".pdf")


    pass