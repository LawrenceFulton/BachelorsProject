import numpy as np
import os
from numpy.core.fromnumeric import argmax, shape 
import pandas as pd
import sys
from sklearn import linear_model
import statsmodels.api as sm
from scipy import stats
from matplotlib import pyplot as plt

def count_majority_vote(array):
    out = []
    out.append( np.count_nonzero(array == -1))
    out.append( np.count_nonzero(array == 0))
    out.append(np.count_nonzero(array == 1))
    major_vote = argmax(out) - 1
    return np.count_nonzero(array == major_vote)




def get_polis_std():

    directory ='../polis/openData'
    sub_dir = next(os.walk(directory))[1]
    out = pd.DataFrame([], columns = ['Dataset name',
                                        'No. of participants',
                                        'No. of comments', 
                                        'No. of votes', 
                                        'std_dev', 
                                        'ratio_accepted',
                                        'vote_of_majority']                                        
                        )

    for sub in sub_dir:
        if sub == '.git':
            continue
        
        complete = directory + "/" + sub
        # print(complete)
        df = pd.read_csv(complete +  "/votes.csv")

        df = df.sort_values(by = 'timestamp')
        df = df.drop(['datetime'], axis = 1)
        df = df.drop(['timestamp'], axis = 1)
        df = df.values

        # print(df)
        len_std = df.shape[0]
        sum_std = 0
        sum_count_of_majority = 0

        for i in range(max(df[:,0])):

            mask = df[:, 0] == int(i)
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
        out = out.append(new_entry)



        print(sub, "no. of participants:", max(df[:,1]), "no of individual comments" , max(df[:,2]), "final_std: ", final_std) 

    out.to_csv("data/polis_std.csv")

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

    
if __name__ == '__main__':
    # get_polis_std()
    regression_polis()
    # plot_data_against_std()