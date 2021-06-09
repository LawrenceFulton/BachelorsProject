import numpy as np
import pandas as pd
import os


def preprossessing(fromPolis = False, onCluster = False, path = ""):
    '''
    preprocesses the data by reducing it to the important stuff and sorting

    :param fromPols bool, true if data shall be taken from the polis dataset
    :param onCluster bool, true if program is used on the cluster
    '''

    if fromPolis:

        if path == "":

            path = "15-per-hour-seattle"
            # path = "american-assembly.bowling-green"
            # path = "brexit-consensus"
            # path = "canadian-electoral-reform"
            # path = "football-concussions"
            # path = "march-on.operation-marchin-orders"
            # path = "scoop-hivemind.affordable-housing"
            # path = "scoop-hivemind.biodiversity"
            # path = "scoop-hivemind.freshwater"
            # path = "scoop-hivemind.ubi"
            # path = "scoop-hivemind.taxes"
            # path = "vtaiwan.uberx"        

        if onCluster:
            df = pd.read_csv("/openData/" + path + "/votes.csv")
        else:
            df = pd.read_csv("../polis/openData/" + path + "/votes.csv")


        df = df.sort_values(by = 'timestamp')
        df = df.drop(['datetime'], axis = 1)
        df = df.drop(['timestamp'], axis = 1)
        
        df = df.values.astype(int)

        if onCluster:
            cmt = pd.read_csv("/openData/" + path + "/comments.csv")
        else:
            cmt = pd.read_csv("../polis/openData/" + path + "/comments.csv")



        ### deleting all moderated comments
        cmt = cmt.loc[cmt['moderated'] == -1]
        bad_cmt_id = cmt['comment-id'].values
        rows_to_delete = []
        for i in range(df.shape[0]):
            cmt_id = df[i,0]
            if cmt_id in bad_cmt_id:
                rows_to_delete.append(i)
        
        cleaned_df = np.delete(df,rows_to_delete, axis= 0)
        print("n_cleaned cmt ", len(rows_to_delete))
        df = cleaned_df



        ### delete all people which aren't included in the clustering 

        if onCluster:
            pts = pd.read_csv("/openData/" + path + "/participants-votes.csv")
        else:
            pts = pd.read_csv("../polis/openData/" + path + "/participants-votes.csv")

        bad_pts_id = pts.loc[pts['n-votes'] <= 5]
        bad_pts_id = bad_pts_id['participant'].values

        rows_to_delete = []
        for i in range(df.shape[0]):
            pts_id = df[i,1]
            if pts_id in bad_pts_id:
                rows_to_delete.append(i)
        
        cleaned_df = np.delete(df,rows_to_delete, axis= 0)
        print("n_cleaned pts ", len(rows_to_delete))
        df = cleaned_df

    else:
        path = "vote_hist_72"
        # path = "vote_hist_backup"

        df = pd.read_csv("data/" + path + '.csv')

        df.columns = ['idx', 'comment-id', 'voter-id', 'vote']
        df = df.drop('idx' ,axis=1)
        df = df.values
        df = df.astype(int)

    return df, path


def get_all_sub_dir():
    directory ='openData'
    sub_dir = next(os.walk(directory))[1]
    return sub_dir[1:]


if __name__ == "__main__":
    get_all_sub_dir()
    