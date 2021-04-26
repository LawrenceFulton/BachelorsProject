import numpy as np
from numpy.linalg import norm
from sklearn.decomposition import PCA
import pandas as pd
import random as rd
import sys


PASS = 0
ACCEPT = 1

def data_creation(n_indi, n_decision, n_proto = 2):
    cutoff = 0.07 # pretty much arbitrarily chosen 

    protos = [0] * n_proto
    
    for i in range(n_proto):
        protos[i] =  np.random.uniform(-1,1,n_decision)
    


    # ################
    # proto1 = np.random.uniform(-1,1,n_decision)
    # proto2 = np.random.uniform(-1,1,n_decision)


    data = np.zeros((n_indi,n_decision))

    for r in range(n_indi):
        believe  = np.random.uniform(0,1,n_proto)
        noise =  np.random.uniform(-0.7,0.7, n_decision)
        
        d = np.zeros(n_decision)

        for p in range(n_proto):
            d += believe[p] * protos[p]


        data[r,:] = d + noise
    



    data = np.where(data < cutoff, data, 1 )
    data = np.where(data > -cutoff, data, -1 )
    data = np.round(data)
    pd.DataFrame(data).to_csv('data/underlying_data.csv')

    print(np.sum(data==1))
    print(np.sum(data==0))
    print(np.sum(data==-1))
    return data

    

def get_distribution(n_indi, n_decision):
    # dist = voting_distribution(n_indi, n_decision)
    # sum_vote = sum(dist)
    # per_dist = dist / sum(dist)
    ratio = n_decision / n_indi
    y = []
    cur_y = n_decision
    for _ in range(n_indi):
        y.append(int(cur_y))
        cur_y -= ratio
    

    
    sum_vote = sum(y)
    per_dist = np.array(y) / sum_vote


    return per_dist, sum_vote 

def priority_metric(A,P,S,E):
    A = np.array(A)
    P = np.array(P)
    S = np.array(S)
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
    return rand_per


def voting_alg(underlying_opinion):
    ## variables 

    # the number of admins 
    n_admins = 2
    # the number of votes the admins create and also vote on
    n_pregiven_votes = underlying_opinion.shape[1] - 1
    
    # the number of participants
    n_participant = underlying_opinion.shape[0]
    
    # the total disussions which can be proposed
    n_votes = underlying_opinion.shape[1]
    
    vote_dist, sum_vote = get_distribution(n_participant, n_votes/4)
    person_list = list(range(n_participant))



    # the votes which are available 
    known_votes = np.zeros([n_participant,n_pregiven_votes])
    # if a user has seen a particular question 
    has_seen = np.zeros([n_participant,n_pregiven_votes],dtype='bool')
    

    # the vote history, which at some point can be imported to continuosConsensus.py
    vote_hist = np.zeros((1,3))

    # Pass / Accepted / Seen, taken from the original closure code, will continuously be updated 
    P = [0] * n_pregiven_votes
    A = [0] * n_pregiven_votes
    S = [n_admins] * n_pregiven_votes


    # init phase
    # a number of admins would add theire own opinions to some questions in the beginning
    # modeling this by hardcoding it. 


    # admins vote on discussions 
    for admin in range(n_admins):
        for vote in range(n_pregiven_votes):

            # if rd.choice([True,False]):
                
            has_seen[admin,vote] = True
            decision = underlying_opinion[admin,vote]
            known_votes[admin,vote] = decision
            if decision == ACCEPT:
                A[vote] += 1
            if decision == PASS:
                P[vote] += 1


            new_item = np.array([vote,admin, decision])
            new_item = new_item.reshape(1,3)

            vote_hist = np.append(vote_hist,new_item, axis=0)
                



    ## only half of the possible votes can be decided on
    for i in range (int(sum_vote)):
        if i % 1000 == 0:
            # update for the cmd 
            print("index", i)
        
        # the number of known people
        n_known_people = known_votes.shape[0]
        # the number of known votes
        n_known_votes = known_votes.shape[1] ## the open question


        ### Choice of person

        # picks a random person from the whole data
        # rand_per = rd.randint(0,n_known_people)
        rand_per = np.random.choice( person_list, 1, p = vote_dist)
        
        ### Choice of question
        
        chosen_question = -99
        # a dynamic probability which will make people propose new questions one in n_known_votes times
        rand_vote = rd.randint(0,n_known_votes)
        if rand_vote == n_known_votes:
            
            # stopping condition so we will stick at the amount of questions people have 
            if rand_vote == n_votes: 
                continue
            # we append the column into the known database 
            question_to_append = np.zeros([n_known_people,1],dtype=bool)
            
            # appending the new empty data to the known data
            known_votes = np.c_[known_votes,question_to_append]
            has_seen = np.c_[has_seen,question_to_append]
    
    
            ## get the actual vote from the underlying knowledgebase
            vote = underlying_opinion[rand_per,rand_vote]
            # updating the has_voted matrix
            has_seen[rand_per,rand_vote] = True

            
            n_known_votes += 1
            chosen_question = rand_vote



            # updating P,S,A
            S.append(1)
            if vote == ACCEPT:
                A.append(1)
                P.append(0)
            elif vote == PASS:
                P.append(1)
                A.append(0)
            elif vote == -1:
                P.append(0)
                A.append(0)
            else:
                print("vote is ", vote)
                print("this should not happen please investigate")
                sys.exit(0)
                # return consensus

        else:

            E = get_e(known_votes)
            priority = priority_metric(A,P,S,E)
            p_has_seen = has_seen[rand_per,:][0] # mask for the current person

            if np.all(p_has_seen):
                # deals with the case that all questions are already seen by this person
                continue

            # cleaning priority so that no question will be proposed which the user has already seen
            cleaned_priority = priority[~p_has_seen]

            if sum(cleaned_priority) == 0:
                continue
            # chosen_question_filtered = get_max(cleaned_priority)
            chosen_question_filtered = get_probability(cleaned_priority)
            # print(chosen_question_filtered)


            # allows us to get back from the cleaned priority to all real values
            chosen_question = -99
            for j in range(len(p_has_seen)):
                if p_has_seen[j] == False:
                    chosen_question_filtered -= 1
                    if chosen_question_filtered == -1:
                        chosen_question = j

            if chosen_question == -99:
                print("sdjabdjkabsdjkabdhasb")
                sys.exit(0)


            vote = underlying_opinion[rand_per, chosen_question]

            known_votes[rand_per,chosen_question] = vote
            has_seen[rand_per,chosen_question] = True

            # updated the A and P values
            if vote == ACCEPT:
                A[chosen_question] += 1
            elif vote == PASS:
                P[chosen_question] += 1 
        
        ## appending the vote into a db which can then be analysed 
        new_item = np.array([chosen_question,rand_per[0], vote[0]])
        new_item = new_item.reshape(1,3)

        vote_hist = np.append(vote_hist,new_item, axis=0)


    ## still have to remove the first row of the hist since it is (0,0,0)
    vote_hist = np.delete(vote_hist, (0), axis=0)



    print("somehow finished:")
    print("known_votes,", known_votes)
    pd.DataFrame(known_votes).to_csv('data/known_votes.csv')
    pd.DataFrame(has_seen).to_csv('data/has_seen.csv')
    pd.DataFrame(vote_hist).to_csv('data/vote_hist.csv')


if __name__ == "__main__":
    n_indi = 70
    n_votes = 540 # number of different votes
    data = data_creation(n_indi, n_votes,5)


    consensus = voting_alg(data)