import numpy as np
from numpy.linalg import norm
from sklearn.decomposition import PCA
import random as rd

class Conversation:
    def priority_metric(A,P,S,E):
        p = (P + 1) / (S + 2)
        a = (A + 1) / (S + 2)

        return ((1 - p) * (E + 1) * a)**2

    def getE(known_votes):
        known_votes_trans = known_votes.T
        reduced_votes = PCA(n_components = 2).fit_transform(known_votes_trans)
        return norm(reduced_votes, axis = 1)


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
            known_votes[admin,vote] = underlying_opinion[admin,vote]

    while(True):
        known_n_people = known_votes.shape[0]
        known_n_votes = known_votes.shape[1]

        rand_per = rd.randint(n_participant)
        
        # if we know a participant already 
        known_person = -1
        values_list = list(participant_dict.values())
        if (rand_per in values_list):
            list_idx =  values_list.index(rand_per)
            known_person = list(participant_dict.keys())[list_idx]
        else:
            ## else we have to add row to the known_votes of form [1,n_votes]
            person_to_append = np.zeros([known_n_votes])
            known_votes = np.r_[known_votes,person_to_append]
            
            # known_votes = np.append(known_votes, [person_to_append], axis = 0) ## added a row for the new person 




             


            




        

    
    print(participant_dict)











    pass




if __name__ == "__main__":
    a = np.array([[1,2,3],[4,11,6],[20,8,9],[10,11,12]])
    # print(a)

    # b = Conversation.getE(a)
    # print(b)

    voting_alg(a)