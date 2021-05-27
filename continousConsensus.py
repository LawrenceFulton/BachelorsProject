import numpy as np
import pandas as pd
from polis_data_processing import count_majority_vote
from matplotlib import pyplot as plt

'''
Takes in the data of one of the openData provided by polis and continuously after each vote updates the consensus 
and creates a plot out of this development.
'''

def preprossessing(fromPolis = True):
    if fromPolis:



        # path = "vtaiwan.uberx"
        path = "march-on.operation-marchin-orders"
        # path = "scoop-hivemind.ubi"
        # path = "scoop-hivemind.taxes"
        # path = "american-assembly.bowling-green"

        df = pd.read_csv("../polis/openData/" + path + "/votes.csv")


        df = df.sort_values(by = 'timestamp')
        df = df.drop(['datetime'], axis = 1)
        df = df.drop(['timestamp'], axis = 1)

    else:
        path = "vote_hist_50"
        # path = "vote_hist_backup"

        df = pd.read_csv("data/" + path + '.csv')

        df.columns = ['idx', 'comment-id', 'voter-id', 'vote']
        df = df.drop('idx' ,axis=1)

    return df.values, path

def cum_mean(df,path):

    sorted_votes = df[:,2]


    print("mean", np.mean(sorted_votes))
    cum_sum = np.cumsum(sorted_votes)
    cum_sum_mean = cum_sum / np.arange(1, len(sorted_votes)+1)

    cum_sum_mean = cum_sum_mean[50:] 


    plt.plot(cum_sum_mean)
    plt.xlabel("votes")
    plt.ylabel("mean of votes")
    plt.savefig("tmp/sum_votes_"+path+"_test.png")
    plt.close()

def cum_std(df, path):
    df = df.astype(int)

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
    plt.ylabel('standard deviation of votes ')
    plt.savefig("figures/out_" + path + "_test2.pdf")


def cum_own_metric(df,path):
    df = df.astype(int)

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
            # arr = n_votes_comment[n_votes_comment > 2]
            
            # n_all_considered_votes = len(arr)

            consensus = sum(majority_comment) / sum(n_votes_comment)#n_all_considered_votes

            out_consensus.append(consensus)

    plt.plot(out_consensus)
    plt.xlabel('votes')
    plt.ylabel('standard deviation of votes ')
    plt.savefig("figures/out_" + path + "_test3.pdf")

    pass


if __name__ == '__main__':
    df, path = preprossessing()
    cum_mean(df,path)
    # cum_std(df,path)
    cum_own_metric(df,path)
    
