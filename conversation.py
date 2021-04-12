import numpy as np
from numpy.core.fromnumeric import cumsum
from numpy.lib.function_base import vectorize
from numpy.linalg import norm
from scipy.sparse.construct import rand
from sklearn import decomposition
from sklearn.decomposition import PCA
import pandas as pd
import random as rd
import sys
from matplotlib import pyplot as plt

PASS = 0
ACCEPT = 1

def data_creation(n_indi, n_decision):
    cutoff = 0.07 # pretty much arbitrarily chosen 

    ################
    proto1 = np.random.uniform(-1,1,n_decision)
    proto2 = np.random.uniform(-1,1,n_decision)


    data = np.zeros((n_indi,n_decision))



    for r in range(n_indi):
        believe  = rd.random()
        # extra_bias =  np.random.uniform(-0.5,0.5, n_decision)
        data[r,:] = (believe  * proto1 + (1 - believe ) * proto2)  #+ extra_bias
	

    # # centering the data of each decision 
    # for c in range(n_decision):
    #     data[:,c] = data[:,c] - np.mean(data[:,c])

    # print(data)



    data = np.where(data < cutoff, data, 1 )
    data = np.where(data > -cutoff, data, -1 )
    data = np.round(data)
    pd.DataFrame(data).to_csv('data/underlying_data.csv')
    # print("mean",np.mean( data))
    # test0 = np.count_nonzero(data == 0)
    # test1 =  np.count_nonzero(data == -1)
    # testminus1 = np.count_nonzero(data == 1)
    # print("after", test0, test1, testminus1)

    return data

def priority_metric(A,P,S,E):
    A = np.array(A)
    P = np.array(P)
    S = np.array(S)
    p = (P + 1) / (S + 2)
    a = (A + 1) / (S + 2)
    return ((1 - p) * (E + 1) * a)**2

def get_e(known_votes):
    known_votes_trans = known_votes.T
    reduced_votes = PCA(n_components = 2).fit_transform(known_votes_trans)
    return norm(reduced_votes, axis = 1)

# calculate the softmax of a vector
def softmax(vector):
    print("vetor",vector)
    e = np.exp(vector)
    print("E: ", e)
    return e / e.sum()

def get_max(vector):
    max_idx = -99
    cur_val = -99
    for i in range(len(vector)):
        if vector[i] > cur_val:
            max_idx = i
            cur_val = vector[i]
    return max_idx

def measuring_consensus(known_votes, has_seen):
    out = 0
    total_votes = np.sum(has_seen) # might have to change so that in only includes if there more than 2 votes in a column 

    for i in range(known_votes.shape[1]):
        votes_i = known_votes[:,i]
        mask_i = has_seen[:,i]
        cleaned_values = votes_i[mask_i]
        std_dev_votes = np.std(cleaned_values)
        weight = np.sum(mask_i)* std_dev_votes / total_votes
        out += weight

    return out

def voting_alg(underlying_opinion):
    # variables 

    # the number of admins 
    n_admins = 5
    # the number of votes the admins create and also vote on
    n_pregiven_votes = int( underlying_opinion.shape[1]  / 2)
    
    # the number of participants
    n_participant = underlying_opinion.shape[0]
    
    # the total disussions which can be proposed
    n_votes = underlying_opinion.shape[1]
    
    # the votes which are available 
    known_votes = np.zeros([n_admins,n_pregiven_votes])
    # if a user has seen a particular question 
    has_seen = np.zeros([n_admins,n_pregiven_votes],dtype='bool')
    

    # the vote history, which at some point can be imported to continuosConsensus.py
    vote_hist = np.zeros((1,3))
    # vote_hist = np.array(3)




    # Pass / Accepted / Seen, taken from the original closure code, will continuously be updated 
    P = [0] * n_pregiven_votes
    A = [0] * n_pregiven_votes
    S = [n_admins]*n_pregiven_votes




    # init phase
    # a number of admins would add theire own opinions to some questions in the beginning
    # modeling this by hardcoding it. 


    # admins vote on discussions 
    for admin in range(n_admins):
        for vote in range(n_pregiven_votes):

            if rd.choice([True,False]):
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
    for i in range (int((n_participant*n_votes) / 2)):
        if i % 1000 == 0:
            # update for the cmd 
            print("index", i)
        
        # the number of known people
        n_known_people = known_votes.shape[0]
        # the number of known votes
        n_known_votes = known_votes.shape[1] ## the open question


        ### Choice of person

        # picks a random person from the whole data
        rand_per = rd.randint(0,n_known_people)

        # if a new person joins 
        if rand_per == n_known_people:

            if rand_per == n_participant:
                # stops that we dont ask more people that we have in the underlying data 
                continue


            person_to_append = np.zeros([1,n_known_votes],dtype='bool')
            known_votes = np.r_[known_votes,person_to_append]
            has_seen = np.r_[has_seen,person_to_append]
            n_known_people += 1

        
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
            # if we choose to answer a question which has previously already been proposed 
            # we have to check if rand_per has already answeed that
            # all_known_votes = np.array(range(n_known_votes))
            # mask = has_seen[rand_per,:]
            # unanswered_votes = all_known_votes[~mask]
            # rand_vote = rd.choice(unanswered_votes)

            E = get_e(known_votes)
            priority = priority_metric(A,P,S,E)
            p_has_seen = has_seen[rand_per,:] # mask for the current person

            if sum(p_has_seen) == len(p_has_seen):
                # deals with the case that all questions are already seen by this person
                continue

            # cleaning priority so that no question will be proposed which the user has already seen
            
            
            cleaned_priority = priority[~p_has_seen]
            # choosing_probability = softmax(cleaned_priority)
            # cum_choosing_probability = cumsum(choosing_probability)
            # r = rd.random()

            # # the chosen question filtered has to still be converted to the real idx
            # chosen_question_filtered = np.argmax(cum_choosing_probability>r) 

            chosen_question_filtered = get_max(cleaned_priority)
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


            '''
            cleaned_priority =  np.where(~p_has_seen,priority,-99 )
            choosing_probability = softmax(cleaned_priority)
            cum_choosing_probability = cumsum(choosing_probability)
            r = rd.random()
            chosen_question = np.argmax(cum_choosing_probability>r)
            '''


            vote = underlying_opinion[rand_per, chosen_question]

            known_votes[rand_per,chosen_question] = vote
            has_seen[rand_per,chosen_question] = True

            # updated the A and P values
            if vote == ACCEPT:
                A[chosen_question] += 1
            elif vote == PASS:
                P[chosen_question] += 1 

        # consensus.append(measuring_consensus(known_votes,has_seen))
        
        ## appending the vote into a db which can then be analysed 
        new_item = np.array([chosen_question,rand_per, vote])
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
    n_indi = 130
    n_votes = 2000 # n_votes_per_person to be frank
    a = data_creation(n_indi, n_votes)


    consensus = voting_alg(a)
    # print(consensus)
    # np.savetxt("consensus.txt", consensus, delimiter=",")
    # plt.plot(consensus)
    # plt.xlabel('votes')
    # plt.ylabel('standard deviation of votes ')
    # plt.savefig("figures/conse.pdf")
