import numpy as np
from numpy.core.fromnumeric import shape
import pandas as pd
from polis_data_processing import count_majority_vote
from matplotlib import pyplot as plt
import preprocessing as pre 
import os

'''
Takes in the data of one of the openData provided by polis and continuously after each vote updates the consensus 
and creates a plot out of this development.
'''

TO_CSV = False



def cum_mean(df,path):

    sorted_votes = df[:,2]

    sorted_votes = (sorted_votes == 1)
    print("mean", np.mean(sorted_votes))
    cum_sum = np.cumsum(sorted_votes)
    cum_sum_mean = cum_sum / np.arange(1, len(sorted_votes)+1)

    start_point = 50
    cum_sum_mean = cum_sum_mean[start_point:] 
    len_votes = len(cum_sum_mean)
    idx = np.arange(start_point, start_point + len_votes)

    if TO_CSV:
        a = pd.DataFrame(columns= ["n_vote", "mean"])
        for i in range(len_votes):
            a = a.append({"n_vote": i, "mean": cum_sum_mean[i] }, ignore_index = True)

        a.to_csv("data/regression_data/mean/"+path+".csv")

    plt.plot(idx, cum_sum_mean)
    plt.xlabel("votes")
    plt.ylabel("mean of votes")
    plt.savefig("figures/mean/"+path+".pdf")
    plt.close()

def cum_std(df, path):

    # number of different questions 
    n_comment = max(df[:,0]) + 1
    print(n_comment)

    # number of votes for each question
    n_votes_comment = np.zeros(int(n_comment))

    # the consensus updated after each time step
    out_consensus = []

    ## array used to keep track of decisions taken 
    restructured_decisions = []

    # creates a list of np arrays which then will include the decisions which are included 
    for i in range(n_comment):
        restructured_decisions.append(np.zeros(0))


    min_votes = 2


    consensus_comment = np.zeros(n_comment)

    # for each row in the sorted df we check the
    for row in df:
        comment_id = row[0]
        vote = row[2]

        n_votes_comment[comment_id] += 1
        votes_on_comment_id = restructured_decisions[comment_id]
        restructured_decisions[comment_id] = np.append(votes_on_comment_id, vote)


        # if there are more than 2 votes for comment x the std gets updated and further investigated
        if n_votes_comment[comment_id] > min_votes:
            std_comment = np.std(restructured_decisions[comment_id])
            consensus_comment[comment_id] = std_comment
        else:
            continue

        consensus = 0
        sum_decisions = sum(n_votes_comment)

        if sum_decisions > 2:
            for i in range(len(n_votes_comment)):
                consensus += n_votes_comment[i] * consensus_comment[i] / sum_decisions

            out_consensus.append(consensus)

    out_consensus = out_consensus[50:]

    plt.plot(out_consensus)
    plt.xlabel('votes')
    plt.ylabel('standard deviation of votes')
    plt.savefig("figures/std/" + path + ".pdf")
    plt.close()


    if TO_CSV:
        len_votes = len(out_consensus)
        a = pd.DataFrame(columns= ["n_vote", "std"])
        for i in range(len_votes):
            a = a.append({"n_vote": i, "std": out_consensus[i] }, ignore_index = True)


        a.to_csv("data/regression_data/std/"+path+".csv")

def cum_own_metric(df,path):

    # number of different questions 
    n_comment = max(df[:,0]) + 1
    print(n_comment)

    # number of votes for each question
    n_votes_comment = np.zeros(int(n_comment))

    # the consensus updated after each time step
    out_consensus = []

    ## array used to keep track of decisions taken 
    restructured_decisions = []

    # creates a list of np arrays which then will include the decisions which are included 
    for _ in range(n_comment):
        restructured_decisions.append(np.zeros(0))


    min_votes = 2
    majority_comment = np.zeros(n_comment)

    # for each row in the sorted df we check the
    for row in df:
        comment_id = row[0]
        vote = row[2]

        n_votes_comment[comment_id] += 1
        votes_on_comment_id = restructured_decisions[comment_id]
        restructured_decisions[comment_id] = np.append(votes_on_comment_id, vote)


        # if there are more than 2 votes for comment x the std gets updated and further investigated
        if n_votes_comment[comment_id] > min_votes:
            n_majority_vote = count_majority_vote(restructured_decisions[comment_id])
            majority_comment[comment_id] = n_majority_vote 
        else:
            continue

        consensus = 0
        sum_decisions = sum(n_votes_comment)

        if sum_decisions > 2:
            arr = n_votes_comment[n_votes_comment > 2]
            
            n_all_considered_votes = sum(arr)

            consensus = sum(majority_comment) / n_all_considered_votes

            out_consensus.append(consensus)

    out_consensus = out_consensus[50:]

    plt.plot(out_consensus)
    plt.xlabel('votes')
    plt.ylabel('metric of votes ')
    plt.savefig("figures/own_metric/" + path + ".pdf")
    plt.close()

    if TO_CSV:
        len_votes = len(out_consensus)
        a = pd.DataFrame(columns= ["n_vote", "own_metric"])
        for i in range(len_votes):
            a = a.append({"n_vote": i, "own_metric": out_consensus[i] }, ignore_index = True)

        a.to_csv("data/regression_data/own_metric/"+path+".csv")



    return out_consensus


def mean_cum_own_metric():

    directory ='../polis/openData'
    sub_dir = next(os.walk(directory))[1]

    all_data = np.zeros([1,15000])

    for sub in sub_dir:
        if sub == '.git':
            continue
        
        complete = directory + "/" + sub
        df = pd.read_csv(complete +  "/votes.csv")            
        df = df.sort_values(by = 'timestamp')
        df = df.drop(['datetime'], axis = 1)
        df = df.drop(['timestamp'], axis = 1)

        df = df.values


        if df.shape[0] > 20000:
            df = df[:20000,:]


        own_data = cum_own_metric(df, sub)
        own_data = np.array(own_data)

        own_data = np.array(own_data)
        
        print("shape before increasing modifying size", own_data.shape[0])


        if own_data.shape[0] < 15000:

            add_lenth = 15000 - own_data.shape[0]
            empty_array = np.empty(add_lenth)
            # empty_array[:] = np.nan
            empty_array[:] = own_data[-1]
            own_data = np.append(own_data, empty_array)
        else:
            print("ELSE")
            own_data = own_data[:15000]

        
        
        own_data = np.reshape(own_data, [1,15000])
        
        print(all_data.shape)

        all_data = np.append(all_data, own_data, axis=0)

        print(all_data)



    all_data = np.delete(all_data, (0), axis = 1)
    print(all_data.shape)

    print(all_data)
    mean_data = np.nanmean(all_data,axis = 0)
    
    print(mean_data.shape)
    plt.plot(mean_data)
    plt.xlabel("votes")
    plt.ylabel("% of people in the majority")
    plt.savefig("tmp/mean_own_metric_all_data.pdf")
    plt.close()
    pass


def analyse_polis():
    sub_dir = pre.get_all_sub_dir()

    for sub in sub_dir:    
        fromPolis = True
        df, path = pre.preprossessing(fromPolis, False, sub)
        # cum_mean(df,path)
        # cum_std(df,path)
        cum_own_metric(df,path)
        # mean_cum_own_metric()



def analyse_own():
    id = 82
    sub_dir = pre.get_all_sub_dir()

    for sub in sub_dir:
        fromPolis = False
        df, path = pre.preprossessing(fromPolis, False, sub)
        # cum_mean(df,path)
        # cum_std(df,path)    
        cum_own_metric(df,path)



if __name__ == '__main__':
    analyse_polis() 
    # analyse_own()

    pass
