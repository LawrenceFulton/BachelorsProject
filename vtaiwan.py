
from numpy.lib.histograms import histogram
from matplotlib import pyplot as plt
import pandas as pd
import questioning_alg as qalg
from scipy.spatial import distance


import numpy as np

path = "vtaiwan.uberx"
# path = "march-on.operation-marchin-orders"

df = pd.read_csv("../polis/openData/" + path + "/participants-votes.csv")
print(df)

## deleting everyone with no group + all not need columns 
df = df.dropna(how= "any", subset=['group-id'])
df = df.drop(['participant'], axis= 1)
df = df.drop(['group-id'], axis= 1)
df = df.drop(['n-comments'], axis= 1)
df = df.drop(['n-votes'], axis= 1)
df = df.drop(['n-agree'], axis= 1)
df = df.drop(['n-disagree'], axis= 1)

print(df)


## Deleting all questions which less than 20 people answered
rows_to_drop = []
min_len = df.shape[0]-20
for i in range(1,df.shape[1]):
    i_str= str(i)
    n_nan = sum(pd.isnull(df[i_str]))
    
    # print("n_nan", n_nan)
    if n_nan > min_len:
        # print("drop"+ i_str)
        rows_to_drop.append(i_str)
        # df = df.drop([i_str], axis= 1)

df = df.drop(rows_to_drop, axis= 1)


data = df.values


print(data)
##########
n_desicion = df.shape[1] # number of desisions each individual has to do 
n_indi = df.shape[0] # the number of individuals 
init_set_size = 2 # the initial set size which each individual has to do 




des_hist = list( range(init_set_size))


## The set of open votes
D = set(range(init_set_size))

## Unknown votes
E = set(range(init_set_size, n_desicion))

## The measurement of consentus 
std_dev = np.nanstd(data,axis=0)

plt.plot(std_dev)
plt.savefig("std_dev")



# ## All the disnces of the data
# trans_data = data.transpose()
# dis_mat = distance.cdist(trans_data, trans_data)

# print(E)
# print(dis_mat)

# while(E):
#     min_sd_idx = qalg.get_min_sd(std_dev, D)
#     min_dis_idx = qalg.get_min_dis(min_sd_idx, dis_mat, E)
#     E.remove(min_dis_idx)
#     D.add(min_dis_idx)
#     des_hist.append(min_dis_idx)

# # print("proto_diff" , proto1-proto2)
# print("std_dev", std_dev)
# print("des_hist" , des_hist)
# out = qalg.rearrang_sd(std_dev, des_hist)

# plt.plot(out,marker = '.')
# plt.savefig("bdas")


