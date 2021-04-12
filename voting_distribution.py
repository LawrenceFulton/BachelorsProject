from matplotlib import pyplot as plt
from numpy.core.fromnumeric import std
import pandas as pd
from scipy import stats
import numpy as np
import os

# # path = "vtaiwan.uberx"
# path = "march-on.operation-marchin-orders"
# # path = "scoop-hivemind.affordable-housing"

# df = pd.read_csv("../polis/openData/" + path + "/participants-votes.csv")


# n_votes = df['n-votes']

# n_votes = n_votes.values

# data = np.sort(n_votes)[::-1]
# x = data / max(data)
# y = np.arange(len(data)) / len(data)


# slope, intercept, r, p, std_err = stats.linregress(x, y)

# def myfunc(x):
#   return slope * x + intercept

# mymodel = list(map(myfunc, x))

# plt.scatter(x, y)
# plt.plot(x, mymodel)
# plt.show()

# print(slope, intercept,r ,p, std_err)

# # print()

# # plt.plot(data)
# plt.savefig("tmp/bla")


directory ='../polis/openData'
sub_dir = next(os.walk(directory))[1]

len_x = list()
len_y = list() 

for sub in sub_dir:
    if sub == '.git':
        continue
    
    complete = directory + "/" + sub
    df = pd.read_csv(complete +  "/participants-votes.csv")

    n_votes = df['n-votes']

    n_votes = n_votes.values

    data = np.sort(n_votes)[::-1]
    len_y.append(max(data))
    len_x.append(len(data))

# print("meanx_x", np.mean(len_x))
# print("std_x", std(len_x))
# print("x,", len_x)
# print("meanx_x", np.mean(len_y))
# print("std_y", std(len_y))

mean_n_participant = np.mean(len_x)
mean_max_votes = np.mean(len_y)
print(mean_n_participant)
print(mean_max_votes)

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
    y = (data / max(data)) * mean_n_participant
    x = (np.arange(len(data)) / len(data)) * mean_max_votes

    all_x = np.append(all_x, x)
    all_y = np.append(all_y, y)




# def myfunc(x):
#   return slope * x + intercept
# slope, intercept, r, p, std_err = stats.linregress(x, y)
# mymodel = list(map(myfunc, all_x))


# plt.scatter(all_x, all_y)
# plt.plot(all_x, mymodel)
# plt.savefig("tmp/bla")
# plt.close()

# #polynomial fit with degree = 2
# model = np.poly1d(np.polyfit(all_x, all_y, 2))

# #add fitted polynomial line to scatterplot
# polyline = np.linspace(1, mean_max_votes, int(mean_max_votes-1))
# plt.scatter(all_x, all_y)
# plt.plot(polyline, model(polyline), c='red')
# plt.savefig("tmp/bla1")

# print(model)


print(all_x)
all_x = all_x.astype(int)

print(all_x)

mean_y = []
for i in range(max(all_x)):
    same_idx = np.where(all_x == i)
    mean_y.append(np.median(all_y[same_idx]))

mean_y = np.array(mean_y)

mean_y = mean_y.astype(int)
print("sum(mean_y)", sum(mean_y))
print(mean_y)

plt.plot(mean_y, c='r')
plt.scatter(all_x, all_y)
plt.savefig("tmp/bla")


np.savetxt("tmp/mean_y.txt", mean_y, delimiter=",")



