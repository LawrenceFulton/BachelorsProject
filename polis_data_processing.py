import numpy as np
import os
from numpy.core.fromnumeric import argmax, shape 
import pandas as pd
import sys
from sklearn import cluster, linear_model
import statsmodels.api as sm
from scipy import stats
from matplotlib import pyplot as plt
import preprocessing as pre
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import pymannkendall as mk





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
            print(file +  ": " , end= "")
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

    plt.boxplot(slope_arr.T)
    plt.savefig("tmp/box")
    plt.close()

    # print("M", np.mean(slope_arr)," SD:" , np.std(slope_arr))
    # t_t = stats.ttest_1samp(slope_arr, 0)
    # print(t_t)
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
    df_o = pd.read_csv('tmp/own_sil_scores_reg.csv', skiprows=0)
    df_o = df_o.drop(df_o.columns[0], axis=1)

    merge_df = pd.merge(df_p, df_o, on=["path", "n_vote"])
    print(merge_df)
    X = merge_df['n_vote']
    Y = merge_df['sil_score_x']
    
    plt.scatter(X,Y)
    plt.savefig("tmp/a.pdf")


    corr, p = stats.pearsonr(X,Y)
    print(corr, p)



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



    
if __name__ == '__main__':
    # get_polis_std()
    # regression_polis()
    # plot_data_against_std()
    # get_polis_ratio_of_votes()
    # get_polis_n_votes()
    # regression_clustering()
    adv_regression_polis()
    # cluster_corr()
    # con_corr("own_metric")