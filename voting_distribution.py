from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def voting_distribution(mean_n_participant = 1193, mean_max_votes = 156 ):
    ''' 
    1193 and 156 are the mean values, and can be found if uncommenting the following codeblock

    len_x == n_participants
    len_y == n_votes
    '''


    directory ='../polis/openData'
    sub_dir = next(os.walk(directory))[1]
    '''
    n_participants = list()
    n_votes = list() 

    
    for sub in sub_dir:
        if sub == '.git':
            continue
        
        complete = directory + "/" + sub
        df = pd.read_csv(complete +  "/participants-votes.csv")

        n_votes_sub = df['n-votes']

        n_votes_sub = n_votes_sub.values

        data = np.sort(n_votes_sub)[::-1]
        n_votes.append(max(data))
        n_participants.append(len(data))



    mean_n_participant = np.mean(n_participants)
    mean_max_votes = np.mean(n_votes)
    '''

    ## next step is to standadise the values, we decide to statndadise them towards the mean value

    l = []
    all_x = np.array(l)
    all_y = np.array([])

    for sub in sub_dir:
        if sub == '.git':
            continue
        
        complete = directory + "/" + sub
        df = pd.read_csv(complete +  "/participants-votes.csv")

        n_votes = df['n-votes']

        n_votes = n_votes.values

        data = np.sort(n_votes)[::-1]
        y = (data / max(data)) * mean_max_votes
        x = (np.arange(len(data)) / len(data)) * mean_n_participant

        all_x = np.append(all_x, x)
        all_y = np.append(all_y, y)


    all_x = all_x.astype(int)


    mean_y = []
    for i in range(max(all_x)):
        same_idx = np.where(all_x == i)
        mean_y.append(np.median(all_y[same_idx]))

    mean_y = np.array(mean_y)
    mean_y = mean_y.astype(int)


    plt.plot(mean_y, c='r')
    plt.scatter(all_x, all_y)
    plt.savefig("tmp/bla")


    # np.savetxt("tmp/mean_y.txt", mean_y, delimiter=",")

    a = (mean_y / sum(mean_y))
    print(sum(a))

    return mean_y


def votes_distribution():
    directory ='../polis/openData'
    sub_dir = next(os.walk(directory))[1]
    
    y = np.array([])
    x = np.array([])

    for sub in sub_dir:
        if sub == '.git':
            continue
        
        complete = directory + "/" + sub
        df = pd.read_csv(complete +  "/comments.csv")

        agree = df['agrees']
        disagree = df['disagrees']



        agree = agree.values
        disagree = disagree.values

        n_votes = agree + disagree

        # data = np.sort(n_votes)[::-1]
        y = np.append(y,n_votes)
        # x.append(len(data))

    y = y[y > 4]
    plt.hist(y)
    plt.savefig("tmp/bla")

    pass


if __name__ == "__main__":
    voting_distribution()
    # votes_distribution()


