from sys import path
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import cluster
import continousConsensus as cc
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import preprocessing as pre
import clustering as cls
import conversation as conv
import mca
import os
import multiprocessing






def cluster_analysis(underlying_data, path, a = None) : 

    cutoff = 5

    un = np.unique(underlying_data[:, 1])
    print("len of unique shiiit ", len(un))


    # the max nr of cmts and pers 
    max_cmt = max(underlying_data[:,0]) + 1
    max_per = max(underlying_data[:,1]) + 1 

    ## data[cmt,per]
    data = np.zeros([max_per, max_cmt])
    has_seen = np.zeros([max_per, max_cmt], dtype= bool)


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
        has_seen[per_id, cmt_id] = True
    

        if (cnt == goal):

                
            bool_cnt_cmt = np.where(cnt_cmt > cutoff, True, False)
            bool_cnt_per = np.where(cnt_per > cutoff, True, False)

            d1 = data[bool_cnt_per,:]
            d2 = d1[:,bool_cnt_cmt]   

            # print("CMT:",sum (bool_cnt_cmt))
            # print("PER:", sum (bool_cnt_per))



            goal += inc
            inc += 1
            if (len(d2) == 0) or (np.unique(d2, axis = 0).shape[0] < 10) or (sum(bool_cnt_per) < 2)  :
                continue
            # print(d2)

            idx.append(cnt)
            # k_labels_2 = rep.k_clustring(d2)
            # k_score_2 = silhouette_score(d2, k_labels_2)
            # k_scores_2.append(k_score_2)

            # agg_labels_2 = rep.agg_clustering(d2)
            # agg_score_2 = silhouette_score(d2, agg_labels_2)
            # # print(agg_score_2)
            # agg_scores_2.append(agg_score_2)
            



            red_data_2 = PCA(n_components=2).fit_transform(d2)
            # print(red_data_2)
            pca_labels_2 = cls.k_clustering(red_data_2)
            try:
                pca_score_2 = cls.silhouette_score(red_data_2, pca_labels_2)
            except:
                print(red_data_2)
                print(pca_labels_2)
                print(np.unique(d2, axis = 0).shape[0] )
                exit()




            pca_labels_3 = cls.k_clustering(red_data_2,3)
            pca_score_3 = cls.silhouette_score(red_data_2, pca_labels_3)
            # pca_scores_3.append(pca_score_3)


            pca_labels_4 = cls.k_clustering(red_data_2,4)
            pca_score_4 = cls.silhouette_score(red_data_2, pca_labels_4)

            pca_scores_2.append(max(pca_score_2, pca_score_3, pca_score_4))


            if type(a) == pd.DataFrame:

                a = a.append({"path":path, "n_vote": cnt, "sil_score": max(pca_score_2, pca_score_3, pca_score_4)}, ignore_index = True)


            # red_data_3 = PCA(n_components=2).fit_transform(d2)
            # pca_labels_3 = cls.k_clustering(red_data_3,3)
            # pca_score_3 = cls.silhouette_score(red_data_3, pca_labels_3)
            # pca_scores_3.append(pca_score_3)






            # labels_3 = rep.clustring(d2,3)
            # score_3 = silhouette_score(d2, labels_3)
            # scores_3.append(score_3)
         
            # labels_4 = rep.clustring(d2,4)
            # score_4 = silhouette_score(d2, labels_4)
            # scores_4.append(score_4)

        if cnt % 1000 == 0:
            # update for the cmd 
            print("index", cnt, " and inc ", inc)
        

        # if cnt > 50000:
        #     break


    bool_cnt_cmt = np.where(cnt_cmt > cutoff, True, False)
    bool_cnt_per = np.where(cnt_per > cutoff, True, False)

    d1 = has_seen[bool_cnt_per,:]
    d2 = d1[:,bool_cnt_cmt]   

    # plt.plot(idx, k_scores_2,  label=id + "K 2")
    # plt.plot(idx, agg_scores_2, label = "agggg")
    plt.plot(idx, pca_scores_2, label = path)
    # plt.plot(idx, pca_scores_3, label = "PCA_3")

    plt.xlabel("Number of votes")
    plt.ylabel("Silhoette scorre")
    # plt.legend()



    # bool_cnt_cmt = np.where(cnt_cmt > cutoff, True, False)
    # bool_cnt_per = np.where(cnt_per > cutoff, True, False)


    # d1 = data[bool_cnt_per,:]
    # d2 = d1[:,bool_cnt_cmt]   
    # red_data = PCA(n_components=2).fit_transform(d2)
    # pca_labels_2 = cls.k_clustering(red_data)
    # pca_score_2 = cls.silhouette_score(red_data, pca_labels_2)
    # pca_scores_2.append(pca_score_2)
    return pca_score_2, a
    pass


def fromPolisData():
    fromPolis = True

    a = pd.DataFrame(columns= ["path", "n_vote", "sil_score"])

    sub_dir = pre.get_all_sub_dir()

    for sub in sub_dir:
        data, path = pre.preprossessing(fromPolis, False, sub)

        score, a = cluster_analysis(data,path, a)


    a.to_csv("tmp/sil_scores_reg.csv")    


    plt.savefig("figures/PCA_cluster/sil_all.pdf")
    plt.close()



def multi_help(i):
    paths = pre.get_all_sub_dir()

    path = 'data/sil_scores/own' +str(i) +'th/'
    try: 
        os.mkdir(path) 
    except OSError as error: 
        print(error) 

    for sd in range(10, 100, 10):
        a = pd.DataFrame(columns= ["path","n_vote", "sil_score"])
    
        for name in paths:
            print(name)
            data = pd.read_csv("data/" + str(i) + "th/" + str(sd) + "/vote_hist_" +  name + ".csv")
            data = data.dropna(axis=0)
            data = data.drop(data.columns[0], axis=1).astype(int).values
            _, a = cluster_analysis(data, name , a)
        
        
        a.to_csv(path + str(sd) + ".csv")




def notFromPolisData():
    pool = multiprocessing.Pool()

    pool.map(multi_help, range(0,10))





def analyse_rd_data():
    global paths 
    paths = pre.get_all_sub_dir()


    for sd in range(60,100, 10):
        a = pd.DataFrame(columns= ["path","n_vote", "sil_score"])
       
        for path in paths:
            print(path)
            data = pd.read_csv("data/random_data/" + str(sd) + "/" +  path + ".csv")
            data = data.drop(data.columns[0], axis=1).astype(int).values
            _, a = cluster_analysis(data, path , a)
        
        
        a.to_csv("data/sil_scores/rd/" + str(sd) + ".csv")



def test():
    fromPolis = False
    all_path = pre.get_all_sub_dir()

    a = []
    for path in all_path:

        data, path = pre.preprossessing(fromPolis, False,path )
        b = cluster_analysis(data,path)
        a.append(b)

    print(np.mean(a))
    print(np.median(a))



if __name__ == '__main__':
    notFromPolisData()
    # fromPolisData()
    # test()
    # data = conv.data_creation(414,112,2, "test")
    
    # data = pd.read_csv("tmp/random_data.csv")
    # data = data.drop(columns=data.columns[0],axis=1)
    # print(data)
    # data = data.astype(int)
    # data = data.values
    # cluster_analysis(data, "test" )




    # red_data_2 = PCA(n_components=2).fit_transform(data)
    # # print(red_data_2)
    # print(red_data_2.shape)
    # pca_labels_2 = cls.k_clustering(red_data_2)
    # pca_score_2 = cls.silhouette_score(red_data_2, pca_labels_2)
    # print(pca_score_2)

    # plt.savefig("tmp/test12.png")

    # analyse_rd_data()
    pass