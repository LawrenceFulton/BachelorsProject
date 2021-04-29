import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

'''
Takes in the data of one of the openData provided by polis and continuously after each vote updates the consensus 
and creates a plot out of this development.
'''
fromPolis = False

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
    path = "vote_hist_46"
    # path = "vote_hist_backup"

    df = pd.read_csv("data/" + path + '.csv')

    df.columns = ['idx', 'comment-id', 'voter-id', 'vote']
    df = df.drop('idx' ,axis=1)



df = df.values


# reducing the matrix size 
# max_size = 10000
# if df.shape[0] > max_size:
    
#     df = df[:max_size,:]


sorted_votes = df[:,2]


print("mean", np.mean(sorted_votes))
cum_sum = np.cumsum(sorted_votes)
cum_sum_mean = cum_sum / np.arange(1, len(sorted_votes)+1)

cum_sum_mean = cum_sum_mean[10:]


plt.plot(cum_sum_mean)
plt.xlabel("votes")
plt.ylabel("mean of votes")
plt.savefig("tmp/sum_votes_"+path)
plt.close()



# print(df)

# df = df.astype(int)

# # number of different questions 
# n_decisions = max(df[:,0]) + 1
# print(n_decisions)

# # number of votes for each question
# n_votes_on_decision = np.zeros(int(n_decisions))

# # the consensus updated after each time step
# out_consensus = []

# ## array used to keep track of decisions taken 
# restructured_decisions = []

# # creates a list of np arrays which then will include the decisions which are included 
# for i in range(n_decisions):
#     restructured_decisions.append(np.zeros(0))


# min_row = 2







# # for each row in the sorted df we check the
# for row in df:
#     comment_id = row[0]
#     voter_id = row[1]
#     vote = row[2]

#     # print(comment_id)
#     n_votes_on_decision[comment_id] += 1 # was voter_id
#     temp_comment_id = restructured_decisions[comment_id]
#     restructured_decisions[comment_id] = np.append(temp_comment_id, vote)

#     n_appearances = []
#     std_dev_per_decision = []
#     ## now looking at how much consensus is already there 

#     for i in range(n_decisions):
#         if(n_votes_on_decision[i] > min_row):
#             deviation = np.std(restructured_decisions[i])
#             std_dev_per_decision.append(deviation)
#             n_appearances.append(restructured_decisions[i])

#     consensus = 0
#     sum_decisions = sum(n_votes_on_decision)

#     if sum_decisions > 2:
#         for i in range(len(n_appearances)):
#             consensus += n_votes_on_decision[i] * std_dev_per_decision[i] / sum_decisions

#         out_consensus.append(consensus)


# test = np.array(out_consensus)

# print(test)
# print("file saved as "+ path)
# test.tofile("data/out_" + path  + ".csv")

# plt.plot(out_consensus)
# plt.xlabel('votes')
# plt.ylabel('standard deviation of votes ')
# plt.savefig("figures/out_" + path + ".pdf")
