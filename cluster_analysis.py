import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import continousConsensus as cc
from sklearn.metrics import silhouette_score
import repness as rep
from sklearn.decomposition import PCA
import preprocessing as pre

import mca






def cluster_analysis(underlying_data, id): 

    cutoff = 5

    un = np.unique(underlying_data[:, 1])
    print("len of unique shiiit ", len(un))


    # the max nr of cmts and pers 
    max_cmt = max(underlying_data[:,0]) + 1
    max_per = max(underlying_data[:,1]) + 1 

    ## data[cmt,per]
    data = np.zeros([max_per, max_cmt])


    bool_cmt = np.zeros(max_cmt, dtype="bool")
    bool_per = np.zeros(max_per, dtype="bool")
    cnt_cmt = np.zeros(max_cmt)
    cnt_per = np.zeros(max_per)

    
    k_scores_2 = []
    scores_3 = []
    scores_4 = []

    agg_scores_2 = []

    pca_scores_2 = []
    pca_scores_3 = []


    idx = []
    cnt = 0
    inc = 10
    goal = 1000

    print("length ", underlying_data.shape[0])

    for row in underlying_data:
        cnt += 1 
        cmt_id = row[0]
        per_id = row[1]
        vote = row[2]

        #ubdate of bool arrays 
        bool_cmt[cmt_id] = True
        bool_per[per_id] = True
        cnt_cmt[cmt_id] += 1
        cnt_per[per_id] += 1


        #update the known data matrix 
        data[per_id, cmt_id] = vote

    
        if (cnt == goal):
                
            bool_cnt_cmt = np.where(cnt_cmt > cutoff, True, False)
            bool_cnt_per = np.where(cnt_per > cutoff, True, False)




            # d1 = data[bool_per,:]
            # d2 = d1[:,bool_cmt]   

            d1 = data[bool_cnt_per,:]
            d2 = d1[:,bool_cnt_cmt]   



            goal += inc
            inc += 1
            
            idx.append(cnt)
            # k_labels_2 = rep.k_clustring(d2)
            # k_score_2 = silhouette_score(d2, k_labels_2)
            # k_scores_2.append(k_score_2)

            # agg_labels_2 = rep.agg_clustering(d2)
            # agg_score_2 = silhouette_score(d2, agg_labels_2)
            # # print(agg_score_2)
            # agg_scores_2.append(agg_score_2)

            red_data = PCA(n_components=2).fit_transform(d2)
            pca_labels_2 = rep.k_clustring(red_data)
            pca_score_2 = rep.silhouette_score(red_data, pca_labels_2)
            pca_scores_2.append(pca_score_2)



            red_data = PCA(n_components=3).fit_transform(d2)
            pca_labels_3 = rep.k_clustring(red_data)
            pca_score_3 = rep.silhouette_score(red_data, pca_labels_3)
            pca_scores_3.append(pca_score_3)

            # labels_3 = rep.clustring(d2,3)
            # score_3 = silhouette_score(d2, labels_3)
            # scores_3.append(score_3)
         
            # labels_4 = rep.clustring(d2,4)
            # score_4 = silhouette_score(d2, labels_4)
            # scores_4.append(score_4)

        if cnt % 1000 == 0:
            # update for the cmd 
            print("index", cnt, " and inc ", inc)
        

        if cnt > 150000:
            break



    # red_data = PCA(n_components=2).fit_transform(d2)
    # labels_2 = rep.k_clustring(red_data)
    # # print("1:", sum(labels_2)," 2:" , len(labels_2)-sum(labels_2))
    # # # s1 = silhouette_score(d2, labels_2)    
    # # # s2 = silhouette_score(red_data, labels_2)    
    # # # print("prepca:", s1, " postpca:",s2)



    # _, best_n = rep.ideal_n_cluster(d2)
    # print("NICE AMOUNT OF CLUSTER K: ", best_n)

    # # _, best_n = rep.ideal_n_cluster(d2, "agg")
    # # print("NICE AMOUNT OF CLUSTER agg: ", best_n)


    # fig, ax = plt.subplots()
    # for g in np.unique(labels_2):
    #     ix = np.where(labels_2 == g)
    #     # if (g == 99 or g == 100):
    #     #     ax.scatter(out[ix, 0], out[ix, 1], s=10, label=g)
    #     # else:
    #     ax.scatter(red_data[ix, 0], red_data[ix, 1], s=1, label=g)

    # plt.legend()
    # plt.savefig("tmp/pca_new")
    # plt.close()
    # pd.DataFrame(d2).to_csv("tmp/d2.csv")

    # pca = PCA(n_components=2, svd_solver='full')
    # pca.fit(d2)
    # print(pca.explained_variance_ratio_)


    print("len per bool", sum(bool_per))
    # # labels_2 = rep.clustring(d2)
   
    # print("K::: 1:", sum(k_labels_2)," 2:" , len(k_labels_2)-sum(k_labels_2))
    # print("AGg::: 1:", sum(agg_labels_2)," 2:" , len(agg_labels_2)-sum(agg_labels_2))

    # plt.plot(idx, k_scores_2,  label=id + "K 2")
    # plt.plot(idx, agg_scores_2, label = "agggg")
    plt.plot(idx, pca_scores_2, label = "PCA_2")
    plt.plot(idx, pca_scores_3, label = "PCA_3")

    ## print(agg_scores_2)


    plt.legend()

    

    pass







if __name__ == '__main__':
    fromPolis = True

    sub_dir = pre.get_all_sub_dir()
    for sub in sub_dir:
        data, path = pre.preprossessing(fromPolis, False, sub)

        cluster_analysis(data,"cleaned_data")
        plt.savefig("figures/PCA_cluster/sil_" + path+ ".pdf")


    pass