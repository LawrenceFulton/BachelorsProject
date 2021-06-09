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



def cum_mean(df,path):

    sorted_votes = df[:,2]


    print("mean", np.mean(sorted_votes))
    cum_sum = np.cumsum(sorted_votes)
    cum_sum_mean = cum_sum / np.arange(1, len(sorted_votes)+1)

    cum_sum_mean = cum_sum_mean[50:] 


    plt.plot(cum_sum_mean)
    plt.xlabel("votes")
    plt.ylabel("mean of votes")
    plt.savefig("figures/cum_mean_"+path+".pdf")
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

    plt.plot(out_consensus)
    plt.xlabel('votes')
    plt.ylabel('percentage of votes in the majority')
    plt.savefig("figures/cum_std_" + path + ".pdf")
    plt.close()

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

    plt.plot(out_consensus)
    plt.xlabel('votes')
    plt.ylabel('metric of votes ')
    plt.savefig("figures/cum_own_metric_" + path + ".pdf")
    plt.close()
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


if __name__ == '__main__':
    fromPolis = True
    df, path = pre.preprossessing()
    cum_mean(df,path)
    cum_std(df,path)
    cum_own_metric(df,path)
    # mean_cum_own_metric()
    
