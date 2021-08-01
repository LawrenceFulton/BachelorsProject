from sys import path
import numpy as np
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import ptp
import pandas as pd
from scipy.stats.stats import mode
from sklearn.decomposition import PCA
import preprocessing as pre
import clustering as cls
import os
import sys


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
    goal = 100

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


            pca_labels_4 = cls.k_clustering(red_data_2,4)
            pca_score_4 = cls.silhouette_score(red_data_2, pca_labels_4)

            pca_scores_2.append(max(pca_score_2, pca_score_3, pca_score_4))


            if type(a) == pd.DataFrame:

                a = a.append({"path":path, "n_vote": cnt, "sil_score": max(pca_score_2, pca_score_3, pca_score_4)}, ignore_index = True)



        if cnt % 1000 == 0:
            # update for the cmd 
            print("index", cnt, " and inc ", inc)
        

        if cnt > 50000:
            break


    bool_cnt_cmt = np.where(cnt_cmt > cutoff, True, False)
    bool_cnt_per = np.where(cnt_per > cutoff, True, False)

    d1 = has_seen[bool_cnt_per,:]
    d2 = d1[:,bool_cnt_cmt]   


    plt.plot(idx, pca_scores_2, label = path)

    plt.xlabel("Number of votes")
    plt.ylabel("Silhoette scorre")
    # plt.legend()



    return pca_score_2, a
    pass


def get_labels(hist: np.array, n_per, n_cmt):
    # print(hist)
    # print(max(hist[:,0]) )
    # max_cmt = max(hist[:,0]) + 1
    # max_per = max(hist[:,1]) + 1 

    ## data[cmt,per]
    data = np.zeros([n_per, n_cmt])

    for row in hist:

        cmt_id = row[0]
        per_id = row[1]
        vote = row[2]
        try:
            data[per_id, cmt_id] = vote
        except:
            print(row)
            sys.exit(0)





    # smoll_data = cls.dimen_reduc(data, 2)
    # labels = cls.k_clustering(smoll_data, 2)
    labels = cls.k_clustering(data, 2)  
    return labels  





def save_labels():

    names = pre.get_all_sub_dir()

    a = pd.read_csv('data/polis_conditions.csv')
    a = a.drop(a.columns[0], axis=1).values

    for i in range(9):

        model_read = "data/model_data/" + str(i) + "th/60/"
        rd_read = "data/random_data/" + str(i) + "th/60/"

        # for name in names:
        for file in a:
            print(file)
            name = file[0]
            n_per = file[2]
            n_cmt = file[1]

            model_data = pd.read_csv(model_read + "vote_hist_"+ name + ".csv")
            model_data = model_data.dropna()
            model_data = model_data.drop(model_data.columns[0], axis=1).values
            model_data = model_data.astype(int)

            rd_data = pd.read_csv(rd_read + name + ".csv")
            rd_data = rd_data.dropna()
            rd_data = rd_data.drop(rd_data.columns[0], axis=1).values
            rd_data = rd_data.astype(int)


            model_labels = get_labels(model_data, n_per, n_cmt)
            rd_labels = get_labels(rd_data, n_per, n_cmt)

            pd.DataFrame(model_labels).to_csv(model_read + "model_labels_"  +name + ".csv" )
            pd.DataFrame(rd_labels).to_csv(rd_read + "rd_labels_"  +name + ".csv" )

            




def fromPolisData():
    fromPolis = True

    a = pd.DataFrame(columns= ["path", "n_vote", "sil_score"])

    sub_dir = pre.get_all_sub_dir()

    for sub in sub_dir:
        data, path = pre.preprossessing(fromPolis, False, sub)

        score, a = cluster_analysis(data,path, a)
        print(score)


    a.to_csv("tmp/sil_scores_reg_1.csv")    


    # plt.savefig("figures/PCA_cluster/sil_all.pdf")
    # plt.close()


def notFromPolisData():
    file_names = pre.get_all_sub_dir()

    for i in range(4,9):

        path = "data/model_data/" + str(i) + "th/"

        # for sd in range(10,100, 10):

        sd = 60 

        a = pd.DataFrame(columns= ["path","n_vote", "sil_score"])
    
        for file in file_names:
            print(file)
            data = pd.read_csv(path + str(sd) + "/vote_hist_" +  file + ".csv")
            data = data.dropna()
            data = data.drop(data.columns[0], axis=1).astype(int).values
            _, a = cluster_analysis(data, file , a)
        
        
        save_path = "data/sil_scores/model/" 
        a.to_csv(save_path + "/60/"+str(i) + "th.csv")

def analyse_rd_data():
    file_names = pre.get_all_sub_dir()
    sd = "60"

    for i in range(4,9):

        path = "data/random_data/" + str(i) + "th/60/"

        # for sd in range(60,100, 10):
        a = pd.DataFrame(columns= ["path","n_vote", "sil_score"])
    
        for file in file_names:
            print(file)
            data = pd.read_csv(path +  file + ".csv")
            data = data.drop(data.columns[0], axis=1).astype(int).values    
            _, a = cluster_analysis(data, file , a)
        
        
        save_path = "data/sil_scores/rd/" 

        try: 
            os.mkdir(save_path) 
        except OSError as error: 
            print(error)  


        a.to_csv(save_path + "60/" +str(i) + "th.csv")



if __name__ == '__main__':
    # notFromPolisData()
    # analyse_rd_data()
    # fromPolisData()
    # test()
    save_labels()
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

    pass