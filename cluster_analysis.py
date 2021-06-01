from repness import prop_pos
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import cluster
import continousConsensus as cc
from sklearn.metrics import silhouette_score
import repness as rep




def cluster_analysis(underlying_data): 
    max_cmt = max(underlying_data[:,0]) +1
    max_per = max(underlying_data[:,1]) +1 
    print(max_cmt, max_per)


    ## data[cmt,per]
    data = np.zeros([max_cmt , max_per])


    bool_cmt = np.zeros(max_cmt, dtype="bool")
    bool_per = np.zeros(max_per, dtype="bool")
    print(bool_cmt)
    counter = 0
    scores_2 = []
    scores_3 = []
    scores_4 = []

    for row in underlying_data:
        counter += 1 
        cmt_id = row[0]
        per_id = row[1]
        vote = row[2]

        #ubdate of bool arrays 
        bool_cmt[cmt_id] = True
        bool_per[per_id] = True


        #update the known data matrix 
        data[cmt_id,per_id] = vote

        d1 = data[bool_cmt,:]
        d2 = d1[:,bool_per]


        if sum(bool_per) > 4: ## we need at least 4 ppl
            labels_2 = rep.clustring(d2)
            score_2 = silhouette_score(d2, labels_2)
            scores_2.append(score_2)

            # labels_3 = rep.clustring(d2,3)
            # score_3 = silhouette_score(d2, labels_3)
            # scores_3.append(score_3)
            
            # labels_4 = rep.clustring(d2,4)
            # score_4 = silhouette_score(d2, labels_4)
            # scores_4.append(score_4)



        if counter > 10000:
            break




    plt.plot(scores_2, label='2')
    # plt.plot(scores_3, label='3')
    # plt.plot(scores_4, label='4')
    plt.legend()

    plt.savefig("tmp/sil_2.png")
    

    pass










if __name__ == '__main__':
    data, path = cc.preprossessing()
    cluster_analysis(data)

    pass