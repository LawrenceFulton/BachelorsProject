import numpy as np
from numpy.core.fromnumeric import cumsum
from numpy.linalg import norm
from sklearn import decomposition
from sklearn.decomposition import PCA
import random as rd

# from sklearn.utils.extmath import softmax
PASS = 0
ACCEPT = 1

def data_creation(n_indi, n_decision):
    # n_decision = 1000 # number of decisions each individual has to do 
    # n_indi = 10000 # the number of individuals 


    ################
    proto1 = np.random.uniform(-1,1,n_decision)
    proto2 = np.random.uniform(-1,1,n_decision)


    data = np.zeros((n_indi,n_decision))



    for r in range(n_indi):
        believe  = rd.random()
        extra_bias =  np.random.uniform(-0.5,0.5, n_decision)
        data[r,:] = (believe  * proto1 + (1 - believe ) * proto2)  + extra_bias
	

    # centering the data of each decision 
    for c in range(n_decision):
        data[:,c] = data[:,c] - np.mean(data[:,c])


    print("data", data)

    data = np.where(data < 0.33, data, 1 )
    data = np.where(data > -0.33, data, -1 )
    data = np.round(data)
    return data


def priority_metric(A,P,S,E):
    p = (P + 1) / (S + 2)
    a = (A + 1) / (S + 2)

    return ((1 - p) * (E + 1) * a)**2

def getE(known_votes):
    known_votes_trans = known_votes.T
    reduced_votes = PCA(n_components = 2).fit_transform(known_votes_trans)
    return norm(reduced_votes, axis = 1)

# calculate the softmax of a vector
def softmax(vector):
    e = np.exp(vector)
    return e / e.sum()


def voting_alg(underlying_opinion):
    # variables 
    n_admins = 2
    n_pregiven_votes = 5
    
    n_participant = underlying_opinion.shape[0]
    n_votes = underlying_opinion.shape[1]
    n_votes_per_participant = np.zeros(n_participant)
    known_votes = np.zeros([n_admins,n_pregiven_votes])
    # joined_participant_id = []

    votes_dict = {}
    participant_dict = {}


    # Pass / Accepted / Seen, taken from the original closure code, will continuously be updated 
    P = [0] * n_pregiven_votes
    A = [0] * n_pregiven_votes
    S = [n_admins]*n_pregiven_votes

    # if a user has seen a particular question 
    has_seen = np.zeros([n_admins,n_pregiven_votes])


    # init phase
    # a number of admins would add theire own opinions to some questions in the beginning
    # modeling this by hardcoding it. 

    for admin in range(n_admins):
        participant_dict.update({admin:admin})
        # joined_participant_id.append(admin)

    for vote in range(n_pregiven_votes):
        votes_dict.update({vote:vote})

    for admin in range(n_admins):
        for vote in range(n_pregiven_votes):
            has_seen[admin,vote] = True
            decision = underlying_opinion[admin,vote]
            known_votes[admin,vote] = decision
            if decision == ACCEPT:
                A[vote] += 1
            if decision == PASS:
                P[vote] += 1

    while(True):
        known_n_people = known_votes.shape[0] ## the n of people already in the known data
        known_n_votes = known_votes.shape[1] ## the open question

        rand_per = rd.randint(n_participant) ## a random person from the whole (unknown) data
        
        known_person = -1
        values_list = list(participant_dict.values())
        # if we know a participant already 
        if (rand_per in values_list):
            list_idx =  values_list.index(rand_per)
            known_person = list(participant_dict.keys())[list_idx]
        else:
            ## else we have to add row to the known_votes of form [1,n_votes]
            person_to_append = np.zeros([known_n_votes])
            known_votes = np.r_[known_votes,person_to_append]
            known_person = known_n_people
            known_n_people += 1
           
        print(known_person)

        # one in n_votes times we let the participant propose a new question 
        if rd.randint(0,known_n_votes) == 0:
            # preparing the known_votes array to be able to get an extra column 
            question_to_append = np.zeros([known_n_people,1])
            known_votes = np.c_[known_votes,question_to_append]

            ## finding an underlying question to ask
            
            all_questions = range(n_votes)
            # all questions which have already been proposed
            question_values_list = list(votes_dict.values())
            # gets the list of questions which haven't been answered
            open_questions = [x for x in all_questions if x not in question_values_list] 
            # chooses one of these not jet proposed question
            proposed_question = rd.choice(open_questions)

            # updating the dictionary
            k = known_n_votes
            votes_dict.update(k,proposed_question)

            ## get the actual vote from the underlying knowledgebase
            vote = underlying_opinion[rand_per,proposed_question]
            known_votes[known_person,k] = vote

            # updating the has_voted matrix
            has_seen[known_person,k] = True


        else:
            E = getE(known_votes)
            priority = priority_metric(A,P,S,E)
            p_has_seen = has_seen[known_person,:]
            # cleaning priority so that no question will be proposed which the user has already seen
            cleaned_priority =  np.where[~p_has_seen,priority,0] 
            choosing_probability = softmax(cleaned_priority)
            cum_choosing_probability = cumsum(choosing_probability)
            r = rd.random()
            proposed_question = np.argmax(cum_choosing_probability>r)
            real_question = votes_dict.get(proposed_question)
            vote = underlying_opinion[rand_per, real_question]
            known_votes[known_person,proposed_question] = vote
            has_seen[known_person,proposed_question] = True



if __name__ == "__main__":
    n_indi = 1000
    n_votes = 1000 # n_votes_per_person to be frank
    a = data_creation(n_indi, n_votes)
    voting_alg(a)