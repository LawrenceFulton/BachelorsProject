from pickle import TRUE
import numpy as np
from numpy.core.fromnumeric import argmax
from numpy.linalg import norm
from sklearn.decomposition import PCA
import pandas as pd
import random as rd
import sys
import test


PASS = 0
ACCEPT = 1

POLIS = 0
EPS = 1


def data_creation(n_indi, n_decision, n_proto = 2, id = ""):
    
    protos = [0] * n_proto
    
    for i in range(n_proto):
        protos[i] =  np.random.uniform(-1,1,n_decision)
    


    # ################
    # proto1 = np.random.uniform(-1,1,n_decision)
    # proto2 = np.random.uniform(-1,1,n_decision)


    data = np.zeros((n_indi,n_decision))

    for r in range(n_indi):
        believe = np.random.uniform(0,1,n_proto)
        believe = believe / sum(believe)

        noise =  np.random.uniform(-0.0001,0.00001, n_decision)
        
        d = np.zeros(n_decision)

        for p in range(n_proto):
            d += believe[p] * protos[p]


        data[r,:] = d + noise
    

    cutoff = 0.1

    data = np.where(data < cutoff, data, 1 )
    data = np.where(data > -cutoff, data, -1 )
    data = np.round(data)
    pd.DataFrame(data).to_csv('data/underlying_data_' +id +'.csv')

    print("1:",np.sum(data==1))
    print("0:",np.sum(data==0))
    print("-1:",np.sum(data==-1))
    return data

def get_distribution(n_indi, n_decision):
    print(n_indi, n_decision)

    ratio = n_decision / n_indi
    y = []
    cur_y = n_decision
    for _ in range(n_indi):
        y.append(int(cur_y))
        cur_y -= ratio
    
    sum_vote = sum(y)
    per_dist = np.array(y) / sum_vote

    print("sum_votes: ", sum_vote)

    return per_dist, sum_vote 

def priority_metric(A,P,S,E):
    '''
    A: Accepted
    P: Passed
    S: Seen
    E: Distance (from center in pca reduced space)
    '''
    p = (P + 1) / (S + 2)
    a = (A + 1) / (S + 2)
    return ((1 - p) * (E + 1) * a)**2

def get_e(known_votes):
    reduced_votes = PCA(n_components = 2).fit_transform(known_votes.T)
    return norm(reduced_votes, axis = 1)

def get_max(vector):
    return vector.argmax(axis=0)

def get_probability(vector):
    enum = np.array(range(len(vector)))
    p = np.array(vector) /sum(vector)
    rand_per = np.random.choice(enum, 1, p = p)
    # rand_per = np.argmax(vector)
    return rand_per

def get_comment_polis(known_votes, P, A, S, has_seen, cur_per ):
    '''
    This is the way comment routing is done by polis
    '''
    E = get_e(known_votes)
    priority = priority_metric(A,P,S,E)
    p_has_seen = has_seen[cur_per,:][0] # mask for the current person

    if np.all(p_has_seen):
        # deals with the case that all questions are already seen by this person
        return -1

    # cleaning priority so that no question will be proposed which the user has already seen
    cleaned_priority = priority[~p_has_seen]

    if sum(cleaned_priority) == 0:
        print("cur_per has open votes", cur_per)
        return -1
    chosen_comment_filtered = get_probability(cleaned_priority)

    # allows us to get back from the cleaned priority to all real values
    chosen_comment = -99
    for j in range(len(p_has_seen)):
        if p_has_seen[j] == False:
            chosen_comment_filtered -= 1
            if chosen_comment_filtered == -1:
                chosen_comment = j

    if chosen_comment == -99:
        print("some mistake has happend")
        sys.exit(0)

    return chosen_comment

def get_comment_eps(has_seen, cur_per, action_value, eps):
    '''
    comment routing based on epsylon greedy 
    '''
    p_has_seen = has_seen[cur_per,:][0]
    if np.all(p_has_seen):
        return -1

    cleaned_action_value = action_value[~p_has_seen]

    if sum(cleaned_action_value) == 0:
        return -1

    if rd.random() > eps:
        chosen_comment_filtered = np.argmax(cleaned_action_value)
    else:
        chosen_comment_filtered = rd.randint(0,len(cleaned_action_value)-1)

    # allows us to get back from the cleaned priority to all real values
    chosen_comment = -99
    for j in range(len(p_has_seen)):
        if p_has_seen[j] == False:
            chosen_comment_filtered -= 1
            if chosen_comment_filtered == -1:
                chosen_comment = j

    if chosen_comment == -99:
        print("some mistake has happend")
        sys.exit(0)

    return chosen_comment

def voting_alg(underlying_opinion, comment_routing, id):
    # the number of participants    
    n_participant = underlying_opinion.shape[0]
    
    # the total disussions which can be proposed
    n_votes = underlying_opinion.shape[1]

    print(n_participant, n_votes)

    # the number of admins 
    n_admins = 2
    # the number of votes the admins create and also vote on
    n_pregiven_votes = n_votes
    
    # vote_dist gives the likellyhood for each person to vote
    # sum_votes gives the total amount of votes we want to investigate
    vote_dist, sum_vote = get_distribution(n_participant-n_admins, n_votes)
    person_list = list(range(n_admins, n_participant))



    # the votes which are available 
    known_votes = np.zeros([n_participant,n_votes])
    # if a user has seen a particular question 
    has_seen = np.zeros([n_participant,n_votes],dtype='bool')
    

    # the vote history, which at some point can be imported to continuosConsensus.py
    vote_hist = np.zeros((1,3))

    # Pass / Accepted / Seen, taken from the original clj code, will continuously be updated 
    P = np.zeros(n_votes)
    A = np.zeros(n_votes)
    S = np.zeros(n_votes)
    
    # needed for EPS
    eps = 0.2
    action_value = np.random.uniform(-0.1, 0.1, n_votes)


    # ranking 
    rank = []
    

    # init phase
    # a number of admins would add theire own opinions to some questions in the beginning
    # modeling this by hardcoding it. 


    # admins vote on discussions 
    for admin in range(n_admins):
        for vote in range(n_pregiven_votes):


            has_seen[admin,vote] = True
            decision = underlying_opinion[admin,vote]
            known_votes[admin,vote] = decision
            if decision == ACCEPT:
                A[vote] += 1
            if decision == PASS:
                P[vote] += 1
            S[vote] += 1


            # new_item = np.array([vote,admin, decision])
            # new_item = new_item.reshape(1,3)

            # vote_hist = np.append(vote_hist,new_item, axis=0)
                


    ## only half of the possible votes can be decided on
    for i in range (int(sum_vote*10/11)):
        if i % 1000 == 0:
            # update for the cmd 
            print("index", i)
        

        ### Choice of person

        # picks a random person from the whole data
        # cur_per = np.random.choice( person_list, 1, p = vote_dist)
        cur_per = np.random.choice( person_list, 1)
        
        ### Choice of question
        
        chosen_comment = -99
        # a dynamic probability which will make people propose new questions one in n_known_votes times

        if comment_routing == POLIS:
            chosen_comment = get_comment_polis(known_votes, P, A, S, has_seen, cur_per)

        elif comment_routing == EPS:
            chosen_comment = get_comment_eps(has_seen, cur_per, action_value, eps )

        else: 
            print("error")
            sys.exit(0)

        if chosen_comment == -1:
            # deals with all errors where one shall skip a question, such as a person has seen each question 
            continue

        vote = underlying_opinion[cur_per, chosen_comment]

        known_votes[cur_per,chosen_comment] = vote
        has_seen[cur_per,chosen_comment] = True

        # updated the A / P / S values
        if vote == ACCEPT:
            A[chosen_comment] += 1
        elif vote == PASS:
            P[chosen_comment] += 1 

        S[chosen_comment] += 1

        if comment_routing == EPS:
            # have to update the action value 
            action_value[chosen_comment] =  1 / (S[chosen_comment]) * (vote - action_value[chosen_comment])
        
        ## appending the vote into a db which can then be analysed 
        new_entry = np.array([chosen_comment, cur_per[0], vote[0]])
        new_entry = new_entry.reshape(1,3)

        vote_hist = np.append(vote_hist, new_entry, axis=0)


        # ## updating ranking 
        # if np.sum(has_seen[:,chosen_comment]) == n_participant:
        #     # print("CHOSEN COMMENT, ", chosen_comment)
        #     # print(has_seen)
        #     rank.append(chosen_comment)



    ## still have to remove the first row of the hist since it is (0,0,0)
    vote_hist = np.delete(vote_hist, (0), axis=0)


    ## saving the data for further analysis
    print("somehow finished:")
    print("known_votes,", known_votes)
    pd.DataFrame(known_votes).to_csv('data/known_votes_'+ id + '.csv')
    pd.DataFrame(has_seen).to_csv('data/has_seen_ '+ id +'.csv')
    pd.DataFrame(vote_hist).to_csv('data/vote_hist_'+ id +'.csv')
    # pd.DataFrame(rank).to_csv('data/rank_'+ id +'.csv')
    # print(rank)





def ran_voting_alg(underlying_opinion, id):
    # the number of participants    
    n_participant = underlying_opinion.shape[0]
    
    # the total disussions which can be proposed
    n_votes = underlying_opinion.shape[1]


    # the votes which are available 
    known_votes = np.zeros([n_participant,n_votes])
    # if a user has seen a particular question 
    has_seen = np.zeros([n_participant,n_votes],dtype='bool')
    

    # the vote history, which at some point can be imported to continuosConsensus.py
    vote_hist = np.zeros((1,3))


    # ranking 
    rank = []
    what = 0
    


    i = 0

    ## only half of the possible votes can be decided on
    while i < (n_participant * n_votes * 10 / 11):
        what = 0
        ### Choice of person

        # picks a random person from the whole data
        cur_per = rd.randint(0, n_participant - 1)
        cur_cmt = rd.randint(0, n_votes -1)
        ### Choice of question
        
        if has_seen[cur_per,cur_cmt] == True:
            what = 1
            continue
        if what == 1:
            return 0
        i += 1

        vote = underlying_opinion[cur_per, cur_cmt]

        known_votes[cur_per,cur_cmt] = vote
        has_seen[cur_per,cur_cmt] = True



        
        ## appending the vote into a db which can then be analysed 
        new_entry = np.array([cur_cmt, cur_per, vote])
        new_entry = new_entry.reshape(1,3)

        vote_hist = np.append(vote_hist, new_entry, axis=0)




    ## still have to remove the first row of the hist since it is (0,0,0)
    vote_hist = np.delete(vote_hist, (0), axis=0)


    ## saving the data for further analysis
    print("somehow finished:")
    print("known_votes,", known_votes)
    pd.DataFrame(known_votes).to_csv('data/known_votes_'+ id + '.csv')
    pd.DataFrame(has_seen).to_csv('data/has_seen_ '+ id +'.csv')
    pd.DataFrame(vote_hist).to_csv('data/vote_hist_'+ id +'.csv')
    # pd.DataFrame(rank).to_csv('data/rank_'+ id +'.csv')
    # print(rank)
    




if __name__ == "__main__":
    id = '72'
    n_indi = 414
    n_votes = 112  # number of different votes
    # data = data_creation(n_indi, n_votes, 2, id)
    data = test.test_data_creation(n_indi, n_votes,2, id)


    consensus = voting_alg(data, POLIS, id)

    # ran_voting_alg(data, "random_selection")