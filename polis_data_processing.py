import numpy as np
import os
from numpy.core.fromnumeric import argmax, shape, size 
import pandas as pd
import sys
from pandas.core.reshape.merge import merge
from sklearn import cluster, linear_model
import statsmodels.api as sm
from scipy import stats
from scipy.spatial import distance
from matplotlib import pyplot as plt
from statsmodels.compat.python import with_metaclass
import preprocessing as pre
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import pymannkendall as mk
import clustering as cls






def count_majority_vote(array):
    out = []
    out.append( np.count_nonzero(array == -1))
    out.append( np.count_nonzero(array == 0))
    out.append(np.count_nonzero(array == 1))
    major_vote = argmax(out) - 1
    return np.count_nonzero(array == major_vote)



def get_polis_std():
    out = pd.DataFrame([], columns = ['Dataset name',
                                        'No. of participants',
                                        'No. of comments', 
                                        'No. of votes', 
                                        'std_dev', 
                                        'ratio_accepted',
                                        'vote_of_majority']                                        
                        )



    sub_dir = pre.get_all_sub_dir()
    for sub in sub_dir:

        # print(complete)
        df,_ = pre.preprossessing(True, True, sub)

        # print(df)
        len_std = df.shape[0]
        sum_std = 0
        sum_count_of_majority = 0

        for i in range(max(df[:,0])):
            # getting a mask of all the comments with id == i
            mask = df[:, 0] == int(i)

            # reducing the dataset given the mask
            small_df = df[mask, :]
            
            len_small_df = small_df.shape[0]


            if len_small_df > 2:

                votes_small_df = small_df[:,2]
                # hist = np.histogram(votes_small_df, bins=3)[0]
                count_of_majority_small_df = count_majority_vote(votes_small_df)

                std_small_df = np.std(small_df[:,2])
                # print(std_small_df)
                sum_std += (len_small_df * std_small_df)
                sum_count_of_majority += count_of_majority_small_df
                
            else:
                len_std -= len_small_df

        final_std = sum_std / len_std
        final_count_of_majority = sum_count_of_majority / len_std

        votes = df[:,2]
        count_of_accept = np.count_nonzero(votes == 1)




        new_entry = pd.DataFrame([[sub, 
                                    max(df[:,1]), 
                                    max(df[:,0]), 
                                    len(votes), 
                                    final_std,
                                    count_of_accept / df.shape[0] ,
                                    final_count_of_majority
                                ]], 
                    columns = ['Dataset name',
                                    'No. of participants',
                                    'No. of comments', 
                                    'No. of votes', 
                                    'std_dev', 
                                    'ratio_accepted',
                                    'vote_of_majority'])
        out = out.append(new_entry, ignore_index=True)



        print(sub, "no. of participants:", max(df[:,1]), "no of individual comments" , max(df[:,2]), "final_std: ", final_std) 

    out.to_csv("data/polis_std.csv")

def get_polis_ratio_of_votes():

    directory ='../polis/openData'
    sub_dir = next(os.walk(directory))[1]

    sum_participants = 0
    sum_votes = 0
    sum_agree = 0
    sum_disagree = 0
    counter = 0

    for sub in sub_dir:
        if sub == '.git':
            continue
            

        counter += 1
        complete = directory + "/" + sub
        # print(complete)
        df = pd.read_csv(complete +  "/participants-votes.csv")    
        n_votes = sum(df['n-votes'])
        n_agree = sum(df['n-agree'])
        n_disagree = sum(df['n-disagree'])

        print(n_votes, n_agree, n_disagree)
        sum_votes += n_votes
        sum_agree += n_agree
        sum_disagree += n_disagree
        sum_participants += len(df['n-votes'])

    print("final")
    print(sum_votes, sum_agree, sum_disagree)

    print("ratio")
    print(sum_votes/counter, sum_agree/counter, sum_disagree/counter)

    print(sum_votes / sum_participants)
    # print("mean_votes_per comment", sum(all_votes)/len(all_votes))


    pass


def get_polis_n_votes(): # number of votes in each discussion
    directory ='../polis/openData'
    sub_dir = next(os.walk(directory))[1]

    sum_n_votes = 0
    votes_per_discussion = []
    counter = 0

    for sub in sub_dir:
        if sub == '.git':
            continue
        
        counter += 1 

        complete = directory + "/" + sub
        # print(complete)
        df = pd.read_csv(complete +  "/votes.csv")    
        df = df.values
        sum_n_votes += df.shape[0]
        votes_per_discussion.append(df.shape[0])
        print(df.shape)

    

    print("meadian ", np.median(votes_per_discussion))
    print("mean  ", sum_n_votes/counter, " votes in a discussion")


    pass    

def regression_polis():
    df = pd.read_csv('data/polis_std.csv')
    print(df)
    column_names = df.columns.values
    df = df.drop(5, axis=0)
    df = df.drop(column_names[:2], axis = 1)
    print(df)
    # df = df.drop('')

    ## normlising data 
    for column in df.columns:
        df[column] = df[column]  / df[column].abs().max()


    # df = df.values
    X = df[['No. of participants','No. of comments','No. of votes']]
    y = df['vote_of_majority']

    regr = linear_model.LinearRegression()
    regr.fit(X, y) 
    print(regr.coef_)


    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    print(est2.summary())

    # print(df)

def plot_data_against_std():
    df = pd.read_csv('data/polis_std.csv')
    column_names = df.columns.values
    df = df.drop(column_names[:2], axis = 1)

    plt.scatter([df['No. of comments']],[df['vote_of_majority']])
    plt.savefig("tmp/scatter_participants_ratio")
    plt.close()

def regression_clustering():

    df = pd.read_csv('tmp/own_sil_scores_reg.csv')

    print(df)
    # normlising data 
    for column in df.columns:
        df[column] = df[column]  / df[column].abs().max()


    # df = df.values
    X = df[["n_vote"]]
    y = df['sil_score']

    regr = linear_model.LinearRegression()
    regr.fit(X, y) 
    print(regr.coef_)


    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    print(est2.summary())


    pass

def adv_regression_polis():
    # metric = "own_metric"

    metric_list = ["std", "mean", "own_metric"]

    slope_arr =np.zeros([3,12])
    cnt_0 = -1
    cnt_1 = -1
    for metric in metric_list:
        cnt_0 += 1
        cnt_1 = -1
        directory ='data/regression_data/' + metric
        # sub_dir = next(os.walk(directory))[1]

        file_names = os.listdir(directory)
        main_df = pd.DataFrame(columns= ["n_vote", metric])

        for file in file_names:
            cnt_1 += 1
            # print(file +  ": " , end= "")
            df = pd.read_csv(directory +"/" + file)
            df = df.drop(df.columns[0], axis=1)

            # df['n_vote'] = df['n_vote'] / df['n_vote'].abs().max()

            df = df.head(35000)

            # main_df = main_df.append(df, ignore_index = True)

            # for column in df.columns:
            #     df[column] = df[column]  / df[column].abs().max()



            # df = df.values
            # X = df[['n_vote']]
            y = df[metric]

            # regr = linear_model.LinearRegression()
            # regr.fit(X, y) 
            # slope_arr[cnt_0, cnt_1] = regr.coef_
            # print(regr.coef_)


            # X2 = sm.add_constant(X)
            # est = sm.OLS(y, X2)
            # est2 = est.fit()
            # print(est2.summary())


            # mod = sm.tsa.arima.ARIMA(y, order=(1, 0, 0))
            # res = mod.fit()
            # print(res.summary())


            # res = adfuller(y)
            # print(res)
            

            # result = mk.original_test(y)
            # print(result)

            result = mk.hamed_rao_modification_test(y)
            # trend, h, p, z, Tau, s, var_s, slope, intercept = mk.hamed_rao_modification_test(data)
            slope = result.slope * len(y)
            # slope_arr.append(slope)
            slope_arr[cnt_0, cnt_1] = slope

            # print(result.trend , slope, result.p)

    print(slope_arr)

    # plt.grid(True)
    plt.boxplot(slope_arr.T, widths= 0.3,
                patch_artist = True,
                notch ='True',                 
                labels = ["Standard Deviation", "% of agreement", "% in majority"])
    plt.ylabel("Slope of notion of consensus")
    plt.savefig("tmp/box_prel.pdf")
    plt.close()

    # print("M", np.mean(slope_arr)," SD:" , np.std(slope_arr))
    t_t = stats.ttest_1samp(slope_arr[0,:], 0)
    print(t_t)
    # cohen_D = np.mean(slope_arr) / np.std(slope_arr)
    # print(cohen_D)


    
    return
        
    df = main_df

    print(df)

    ## normlising data 
    # for column in df.columns:
    #     df[column] = df[column]  / df[column].abs().max()



    # df = df.values
    X = df[['n_vote']]
    y = df[metric]

    # print("LENGTH OF ALL:" , len(y))

    # plt.scatter(X,y)
    # plt.xlabel("votes")
    # plt.ylabel("std")
    # plt.savefig("tmp/scatter_all_data")


    regr = linear_model.LinearRegression()
    regr.fit(X, y) 
    print(regr.coef_)


    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    print(est2.summary())




    pass

def cluster_corr():
    df_p = pd.read_csv('tmp/sil_scores_reg.csv')
    df_p = df_p.drop(df_p.columns[0], axis=1)
    # df_o = pd.read_csv('tmp/own_sil_scores_reg.csv', skiprows=0)


    merge_df = pd.DataFrame(columns=["path",  "n_vote",  "sil_score_x" , "sil_score_y" , "sil_score"])

    for i in range(0, 9):

        df_r = pd.read_csv('data/sil_scores/rd/60/' + str(i) + 'th.csv')
        df_r = df_r.drop(df_r.columns[0], axis=1)

        # print(df_r)

        df_o =  pd.read_csv('data/sil_scores/model/60/' + str(i) + 'th.csv')
        df_o = df_o.drop(df_o.columns[0], axis=1)

        # print(df_o)

        new_merge_df = pd.merge(df_p, df_r, on=["path", "n_vote"])
        # print(merge_df)
        new_merge_df = pd.merge(new_merge_df, df_o, on=["path", "n_vote"])
        merge_df = merge_df.append(new_merge_df , ignore_index=True)

    print(merge_df)
    X = merge_df['sil_score_x']
    Y = merge_df['sil_score_y']
    Y1 = merge_df['sil_score']
    merge_df.to_csv("tmp/mergedf.csv")


    plt.scatter(X,Y1, s= 1)
    plt.xlabel("Given data")
    plt.ylabel("Created data")
    plt.savefig("tmp/scatter_" + str(i) + ".png")
    plt.close()


    corr, p = stats.pearsonr(X,Y)
    print("RD:" , i , corr, p)

    corr, p = stats.pearsonr(X,Y1)
    print("own:" , i , corr, p)

def con_corr(metric):

    sub_dir = pre.get_all_sub_dir()

    x_metric = metric + "_x"
    y_metric = metric + "_y"

    a = pd.DataFrame(columns=['n_vote', x_metric, 'set', y_metric])

    for dir in sub_dir:
        df_p = pd.read_csv('data/regression_data/'+metric+'/'+dir+'.csv')
        df_p = df_p.drop(df_p.columns[0], axis=1)

        df_o = pd.read_csv('data/own_regression_data/'+ metric+'/'+dir+'.csv')
        df_o = df_o.drop(df_o.columns[0], axis=1)

        df_p['set'] = dir
        merge_df = pd.merge(df_p, df_o, on=["n_vote"])
        # print(merge_df)
        a = a.append(merge_df)
    print(a)
    X = a[x_metric]
    Y = a[y_metric]
    
    corr, p = stats.pearsonr(X,Y)
    print(corr, p)

    plt.scatter(X,Y,s=1,marker= ",")
    plt.savefig("tmp/a.pdf")

    # a.to_csv("tmp/all_rows.csv")

def cluster_score_evaluation():
    df_p = pd.read_csv('data/sil_scores/real/sil_scores_reg.csv')
    df_p = df_p.drop(df_p.columns[0], axis=1)

    p_groupby = df_p.groupby(["path"])["sil_score"].mean().values
    print(p_groupby)


    # all_r = pd.DataFrame()
    all_r = pd.DataFrame(columns= ["path","n_vote", "sil_score", "trial"])

    all_m = pd.DataFrame(columns= ["path","n_vote", "sil_score", "trial"])


    for i in range(0, 9):

        df_r = pd.read_csv('data/sil_scores/rd/60/' + str(i) + 'th.csv')
        df_r = df_r.drop(df_r.columns[0], axis=1)
        df_r["trial"] = int(i)
        all_r = all_r.append(df_r, ignore_index=True)

        df_m =  pd.read_csv('data/sil_scores/model/60/' + str(i) + 'th.csv')
        df_m = df_m.drop(df_m.columns[0], axis=1)
        df_m["trial"] = int(i)
        all_m = all_m.append(df_m, ignore_index=True)

        


    r_groupby = all_r.groupby(["path", "trial"])["sil_score"].mean().values
    print(len(r_groupby))

    m_groupby = all_m.groupby(["path", "trial"])["sil_score"].mean().values
    print(len(m_groupby))

    t = stats.ttest_rel(r_groupby, m_groupby)

    print("TTEST: ", t)

    print(stats.shapiro(r_groupby))
    print(stats.shapiro(m_groupby))


    plt.boxplot([p_groupby,m_groupby, r_groupby], labels=[ "polis", "model", "random"])
    plt.savefig("tmp/box.png")
    plt.close()

    print(np.mean(r_groupby))
    print(np.mean(m_groupby))
    print(t)


    print(p_groupby)
    b = list(p_groupby)
    b = b * int((len(m_groupby) / 12))


    print(b)
    corr, p = stats.pearsonr(b,r_groupby)
    print("RD:" , i , corr, p)

    corr, p = stats.pearsonr(b,m_groupby)
    print("own:" , i , corr, p)




    #     merge_df = pd.merge(df_p, df_r, on=["path", "n_vote"])
    #     # print(merge_df)
    #     merge_df = pd.merge(merge_df, df_o, on=["path", "n_vote"])
    #     # print(merge_df)
    #     X = merge_df['sil_score_x']
    #     Y = merge_df['sil_score_y']
    #     Y1 = merge_df['sil_score']

    #     plt.scatter(X,Y1, s= 1)
    #     plt.xlabel("Given data")
    #     plt.ylabel("Created data")
    #     plt.savefig("tmp/scatter_" + str(i) + ".png")
    #     plt.close()


    #     corr, p = stats.pearsonr(X,Y)
    #     print("RD:" , i , corr, p)

    #     corr, p = stats.pearsonr(X,Y1)
    #     print("own:" , i , corr, p)



    pass

def cluster_score_evaluation_new():
    a = pd.read_csv('data/polis_conditions.csv')
    a = a.drop(a.columns[0], axis=1).values

    data_names = a[:,0]
    
    df_p = pd.read_csv('data/sil_scores/real/sil_scores_reg.csv')
    df_p = df_p.drop(df_p.columns[0], axis=1)

    bla = []

    for name in data_names:
        # if df_r['path'] is name:
        new_p = df_p.where(df_p['path'] == str(name))
        new_p = new_p.dropna()
        x_p = new_p['n_vote'].values
        x_p = np.arange(0,len(x_p))
        x_p = x_p.reshape(-1,1)            
        # x_r = x_r.reshape(len(x_r), 1).reshape(-1,1)
        y_p = new_p['sil_score'].values
        model_r = linear_model.LinearRegression().fit(x_p, y_p)             
        bla.append(model_r.coef_[0] )


    print(bla)

    slopes = np.zeros([2, 12 * 9])

    for i in range(0, 9):

        df_r = pd.read_csv('data/sil_scores/rd/60/' + str(i) + 'th.csv')
        df_r = df_r.drop(df_r.columns[0], axis=1)

        df_m = pd.read_csv('data/sil_scores/model/60/' + str(i) + 'th.csv')
        df_m = df_m.drop(df_m.columns[0], axis=1)        

        counter = 0
        for name in data_names:
            # if df_r['path'] is name:
            new_r = df_r.where(df_r['path'] == str(name))
            new_r = new_r.dropna()
            x_r = new_r['n_vote'].values
            x_r = np.arange(0,len(x_r))
            x_r = x_r.reshape(-1,1)            
            # x_r = x_r.reshape(len(x_r), 1).reshape(-1,1)
            y_r = new_r['sil_score'].values
            model_r = linear_model.LinearRegression().fit(x_r, y_r)             
            slopes[0, 12*i + counter] = model_r.coef_ 

            new_m = df_m.where(df_m['path'] == str(name))
            new_m = new_m.dropna()
            x_m = new_m['n_vote'].values

            x_m = np.arange(0,len(x_m))
            x_m = x_m.reshape(-1,1)
            # x_m = x_m.reshape(len(x_m), 1).reshape(-1,1)
            y_m = new_m['sil_score'].values
            model_m = linear_model.LinearRegression().fit(x_m, y_m)             
            slopes[1, 12*i + counter] = model_m.coef_ 

            counter += 1
        
    print(slopes[0])

    t = stats.ttest_rel(slopes[0], slopes[1])

    print("TTEST: ", t)

    print(stats.shapiro(slopes[0]))
    print(stats.shapiro(slopes[1]))


    plt.boxplot([slopes[0], slopes[1]],
                patch_artist = True,
                notch ='True', 
                widths= 0.3,
                labels=[ "random model", "polis model"])
    plt.ylabel("Slope of silhouette score")
    plt.savefig("tmp/boxing_day.pdf")
    plt.close()

    print(np.mean(slopes[0]))
    print(np.mean(slopes[1]))


    cohen_D = np.mean(slopes[0] - slopes[1]) / np.std(slopes)
    print("DDDD: ", cohen_D)



    # print(p_groupby)
    # b = list(p_groupby)
    # b = b * int((len(m_groupby) / 12))


    # print(b)
    # corr, p = stats.pearsonr(b,r_groupby)
    # print("RD:" , i , corr, p)

    # corr, p = stats.pearsonr(b,m_groupby)
    # print("own:" , i , corr, p)




    #     merge_df = pd.merge(df_p, df_r, on=["path", "n_vote"])
    #     # print(merge_df)
    #     merge_df = pd.merge(merge_df, df_o, on=["path", "n_vote"])
    #     # print(merge_df)
    #     X = merge_df['sil_score_x']
    #     Y = merge_df['sil_score_y']
    #     Y1 = merge_df['sil_score']

    #     plt.scatter(X,Y1, s= 1)
    #     plt.xlabel("Given data")
    #     plt.ylabel("Created data")
    #     plt.savefig("tmp/scatter_" + str(i) + ".png")
    #     plt.close()


    #     corr, p = stats.pearsonr(X,Y)
    #     print("RD:" , i , corr, p)

    #     corr, p = stats.pearsonr(X,Y1)
    #     print("own:" , i , corr, p)



    pass

def get_under_labels():
    a = pd.read_csv('data/polis_conditions.csv')
    a = a.drop(a.columns[0], axis=1).values
    # print(a)
    data_names = a[:,0]
    
    # read_path = 'data/'
    for i in range(9):
        read_path = 'data/model_data/' + str(i) + "th/60/underlying_data_"
        save_path = 'data/model_data/' + str(i) + "th/60/labels_"
        for name in data_names:

            under_data = pd.read_csv(read_path + name + ".csv")
            under_data = under_data.drop(under_data.columns[0], axis=1)
            
            
            # smoll_data = cls.dimen_reduc(under_data, 2)
            # labels = cls.k_clustering(smoll_data, 2)

            labels = cls.k_clustering(under_data, 2)


            # score = cls.silhouette_score(smoll_data, labels)
            pd.DataFrame(labels).to_csv(save_path + name + ".csv")





    

    pass


def compare_labels():
    names = pre.get_all_sub_dir()

    dist = np.zeros([2,9,12])


    for i in range(3): ######### has to be changed to the number of repetitions
        under_read = "data/model_data_new/" + str(i) + "th/60/"
        model_read = "data/model_data/" + str(i) + "th/60/"
        rd_read = "data/random_data/" + str(i) + "th/60/"


        cnt = -1
        for name in names:
            cnt += 1


            under_lbs =  pd.read_csv(under_read + "cluster_" + name +  ".csv")
            under_lbs = under_lbs.dropna()
            under_lbs = under_lbs.drop(under_lbs.columns[0], axis=1).values
            under_lbs = under_lbs.astype(int).reshape(1,-1)

            model_lbs  = pd.read_csv(model_read + "model_labels_" + name +  ".csv")
            model_lbs = model_lbs.dropna()
            model_lbs = model_lbs.drop(model_lbs.columns[0], axis=1).values
            model_lbs = model_lbs.astype(int).reshape(1,-1)

            rd_lbs  = pd.read_csv(rd_read + "rd_labels_" + name  +  ".csv")
            rd_lbs = rd_lbs.dropna()
            rd_lbs = rd_lbs.drop(rd_lbs.columns[0], axis=1).values
            rd_lbs = rd_lbs.astype(int).reshape(1,-1)


            dist_model_1 = distance.jaccard(under_lbs, model_lbs)
            dist_model_2 = distance.jaccard(under_lbs, (model_lbs) - 1 * -1)
            
            dist_rd_1 = distance.jaccard(under_lbs, rd_lbs)
            dist_rd_2 = distance.jaccard(under_lbs, (rd_lbs) - 1 * -1)


            dist_model = min(dist_model_1, dist_model_2)
            dist_rd = min(dist_rd_1, dist_rd_2)


            # print(dist_model)
            # print(dist_rd)
            # print("_________________")



            # print(under_lbs)
            # print(model_lbs)
            # print(rd_lbs)
            # print((rd_lbs.shape == model_lbs.shape) and (model_lbs.shape == under_lbs.shape))
            dist[0,i,cnt] = dist_model
            dist[1,i,cnt] = dist_rd


    



    mean_dist = np.mean(dist, axis = 1)
    print(mean_dist)
    print("MEAN ", np.mean(mean_dist,axis=1))


    plt.boxplot([mean_dist[0,:], mean_dist[1,:]],labels= ["model", "random"])
    plt.savefig("tmp/aaa.png")
    plt.close()

    t = stats.ttest_rel(mean_dist[0,:], mean_dist[1,:])
    print(t)



    print(mean_dist)





    
if __name__ == '__main__':
    # get_polis_std()
    # regression_polis()
    # plot_data_against_std()
    # get_polis_ratio_of_votes()
    # get_polis_n_votes()
    # regression_clustering()
    # adv_regression_polis()
    # cluster_corr()
    # con_corr("own_metric")
    # cluster_score_evaluation()
    # cluster_score_evaluation_new()
    # get_under_labels()
    compare_labels()