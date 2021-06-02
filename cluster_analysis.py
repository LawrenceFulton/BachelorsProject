from numpy.core.fromnumeric import shape
from conversation import data_creation
from repness import prop_pos
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import cluster
import continousConsensus as cc
from sklearn.metrics import silhouette_score
import repness as rep

def fetch_data():


    df = pd.read_csv("data/votes.csv")
    print(df)

    df = df.sort_values(by = 'timestamp')
    df = df.drop(['datetime'], axis = 1)
    df = df.drop(['timestamp'], axis = 1)

    df = df.values
    return df


def clean_data(df, path):
    '''
    Cleans the data from all the comments which are moderated
    '''

    # cmt = pd.read_csv("data/comments.csv")
    cmt = pd.read_csv("../polis/openData/" + path + "/comments.csv")

    cmt = cmt.loc[cmt['moderated'] == -1]
    bad_cmt_id = cmt['comment-id'].values
    rows_to_delete = []
    for i in range(df.shape[0]):
        cmt_id = df[i,0]
        if cmt_id in bad_cmt_id:
            rows_to_delete.append(i)
    
    cleaned_df = np.delete(df,bad_cmt_id, axis= 0)
    print("n_cleaned cmt ", len(bad_cmt_id))

    return cleaned_df



def cluster_analysis(underlying_data, id): 
    print(underlying_data)


    max_cmt = max(underlying_data[:,0]) +1
    max_per = max(underlying_data[:,1]) +1 
    print(max_cmt, max_per)


    ## data[cmt,per]
    data = np.zeros([max_cmt , max_per])


    bool_cmt = np.zeros(max_cmt, dtype="bool")
    bool_per = np.zeros(max_per, dtype="bool")
    counter = 0
    scores_2 = []
    scores_3 = []
    scores_4 = []
    idx = []


    print(underlying_data.shape)
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


        if counter % 10 == 0 and counter != 0:
            idx.append(counter)
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

        if counter % 100 == 0:
            # update for the cmd 
            print("index", counter)
        

        if counter > 10000:
            break


    plt.plot(idx,scores_2,  label=id + "2")
    plt.plot(idx, scores_3, label=id + "3")
    # plt.plot(scores_4, label='4')
    plt.legend()

    

    pass










if __name__ == '__main__':
    data, path = cc.preprossessing(True)
    c_data = clean_data(data, path)

    cluster_analysis(data,"dirt")
    cluster_analysis(c_data, "clean")
    plt.savefig("tmp/sil_cldasbkdsabd.png")


    pass