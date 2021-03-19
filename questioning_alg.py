import numpy as np
import random as rd
from numpy.core.fromnumeric import std
from numpy.core.numeric import Inf
from matplotlib import pyplot as plt
from scipy.spatial import distance
from sklearn import decomposition
from sklearn.cluster import KMeans




def get_min_sd(std_dev, D):
	'''
	returns the smallest std for the 
	set of desicions already known, 
	thus the most agreed opionion
	'''
	min_idx = -1
	min_val = Inf

	for i in D:
		if min_val > std_dev[i] :
			min_val = std_dev[i]
			min_idx = i

	return min_idx

def get_min_dis(min_sd, dis_mat, E):
	'''
	returns the desicion closest to the one with
	the lowest sd (most agreed one) and hastn't yet 
	been asked
	'''
	min_dis = Inf
	min_idx = -1

	for i in E:

		if min_dis > dis_mat[min_sd, i]:
			min_idx = i
			min_dis = dis_mat[min_sd,i]

	return min_idx

def rearrang_sd(std_dev, des_hist):
	'''
	Needed for the plotting of the sd of the 
	desisions vs when they were asked
	'''
	out = []
	for i in des_hist:
		out.append(std_dev[i])

	return out

def plot_consentus(data):
	kmenas = KMeans(n_clusters=2).fit(data)
	out = decomposition.PCA(n_components=2).fit_transform(data)



	fig, ax = plt.subplots()
	for g in np.unique(kmenas.labels_):
		ix = np.where(kmenas.labels_ == g)
		ax.scatter(out[ix, 0], out[ix, 1], s=1, label=g)
	plt.savefig("figures/pca2")



	# plt.scatter(out[:,0],out[:,1])
	# plt.savefig("figures/pca")
	plt.close()



def main():
	# rd.seed(0)

	##########
	n_desicion = 1000 # number of desisions each individual has to do 
	n_indi = 10000 # the number of individuals 
	init_set_size = 1 # the initial set size which each individual has to do 


	################
	proto1 = np.random.uniform(-1,1,n_desicion)
	proto2 = np.random.uniform(-1,1,n_desicion)


	data = np.zeros((n_indi,n_desicion))



	for r in range(n_indi):
		belive = rd.random()
		extra_bias =  np.random.uniform(-0.5,0.5, n_desicion)
		data[r,:] = (belive * proto1 + (1 - belive) * proto2)  + extra_bias
	

	# centering the data of each desicion 
	for c in range(n_desicion):
		data[:,c] = data[:,c] - np.mean(data[:,c])


	print("data", data)

	data = np.where(data < 0.33, data, 1 )
	data = np.where(data > -0.33, data, -1 )
	data = np.round(data)

	plot_consentus(data)


	## creating histoy of what has been decided on 
	des_hist = list( range(init_set_size))


	## The set of open votes
	D = set(range(init_set_size))

	## Unknown votes
	E = set(range(init_set_size, n_desicion))

	## The measurement of consentus 
	std_dev = np.std(data,axis=0)

	## All the disnces of the data
	trans_data = data.transpose()
	dis_mat = distance.cdist(trans_data, trans_data)



	while(E):
		min_sd_idx = get_min_sd(std_dev, D)
		min_dis_idx = get_min_dis(min_sd_idx, dis_mat, E)
		E.remove(min_dis_idx)
		D.add(min_dis_idx)
		des_hist.append(min_dis_idx)

	print("proto_diff" , proto1-proto2)
	print("std_dev", std_dev)
	print("des_hist" , des_hist)
	out = rearrang_sd(std_dev, des_hist)

	plt.plot(out)
	plt.savefig("figures/question_alg.pdf")
	plt.close()








if __name__ == "__main__":
	main()