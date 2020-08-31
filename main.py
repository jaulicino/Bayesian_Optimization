#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import numpy as np
from matplotlib import pyplot as plt
import sklearn as sk
from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import catch_warnings
from warnings import simplefilter
import scipy
from logP import *
from get_pyrosetta_scores import get_pyrosetta_scores_from_sequence
from bmDCA import *
from matplotlib import pyplot as plt
import gc
import time


# In[2]:


def hammingDist(str1, str2): 
	count = 0 
	for i in range(len(str1)):
		if(str2[i] != str1[i]):
			count+=1
	return count/len(str1)


# In[3]:


# scalarization of objectives (y)
# sum of coefficients equals one
# will minimize "cost"
# calculate scalarized cost for a single y value
vae = VAE_Operator()

def multi_dimensional_score(x):
	sequence = vae.get_seq_from_z(x)
	vae_prob = vae.get_score_from_z(x)[0] 
	pyr_score = get_pyrosetta_scores_from_sequence(sequence) * (-1)
	dca = DCA_scorer()
	dca_prob = dca.get_dca_scores_from_seq(sequence)[0] 
	functionality_score = hammingDist("A"*62, sequence)
	return [vae_prob, pyr_score, dca_prob, functionality_score]

def scalarized_cost(y_singular,lambda_,rho=0.05):
	lambda_ /= np.sum(lambda_)
	p = np.multiply(y_singular, lambda_)
	return np.max(p) + np.sum(p * rho)

#calculate cost for the entire field of observations
def calc_scal_from_observations(y):
	lambda_ = np.random.random(len(y[0]))
	lambda_ /= np.sum(lambda_)
	scalar_scores = np.zeros(len(y))
	for counter, y_singular in enumerate(y):
		cost = scalarized_cost(y_singular, lambda_)
		scalar_scores[counter] = cost
	return (scalar_scores, lambda_)

#expected improvement (to be maximized across whole x_space)
def EI(muNew, stdNew, fMax, epsilon=0.01):
	"""
	Expected improvement acquisition function
	INPUT:
	- muNew: mean of predicted point in grid
	- stdNew: sigma (square root of variance) of predicted point in grid
	- fMax: observed or predicted maximum value (depending on noise p.19 Brochu et al. 2010)
	- epsilon: trade-off parameter (>=0)
	[Lizotte 2008] suggest setting epsilon = 0.01 (scaled by the signal variance if necessary)  (p.14 [Brochu et al. 2010])
	OUTPUT:
	- EI: expected improvement for candidate point
	As describend in:
	E Brochu, VM Cora, & N de Freitas (2010):
	A Tutorial on Bayesian Optimization of Expensive Cost Functions, with Application to Active User Modeling and Hierarchical Reinforcement Learning,
	arXiv:1012.2599, http://arxiv.org/abs/1012.2599.
	"""

	Z = (muNew - fMax - epsilon) / stdNew

	return (muNew - fMax - epsilon) * scipy.stats.norm.cdf(
		Z
	) + stdNew * scipy.stats.norm.pdf(Z) * (-1)


# In[4]:


class BO():
	def __init__(self, initial_x, initial_y):
		self.model = GaussianProcessRegressor()
		self.observed_x = initial_x
		self.observed_y = initial_y
		self.observed_y_scalarized, _ = calc_scal_from_observations(self.observed_y)
		self.model.fit(self.observed_x, self.observed_y_scalarized)
		self.average_scores = np.zeros(1)

	def propose_new(self, resolution=10000000, dim=3):
		best = np.max(self.observed_y_scalarized)
		x_space = np.random.randn(resolution,dim)#generate a bunch of random sequences to maximize acq
		mu, std = self.model.predict(x_space, return_std = True)
		#evaluate ei at a bunch of points in x_space
		scores = EI(mu, std, best)
		#maximize ei
		max_i = np.argmax(scores) 
		new_point = x_space[max_i]
		print(new_point, "-suggested new point")
		return new_point
	
	
	def run_BO(self, num_iter=100):
		start_t = time.time()
		time.clock()    
		self.average_scores = np.zeros((num_iter, len(self.observed_y[0])))
		for num_round in range(num_iter):
			#calculate random scalarization for this round
			self.observed_y_scalarized, lambda_ = calc_scal_from_observations(self.observed_y)
			#fit GPR to new random scalarization
			self.model.fit(self.observed_x, self.observed_y_scalarized)
			#plt.show()
			print("Round ", num_round, " mean score: ", np.mean(self.observed_y_scalarized),flush=True)
			print("Random Scalarization (Lambda): ", lambda_, flush=True)
			
			new_point_to_check = self.propose_new()
			self.observed_x = np.vstack([self.observed_x, new_point_to_check])
			self.observed_y = np.vstack([self.observed_y, multi_dimensional_score(new_point_to_check)])
			
			for i in range(len(self.observed_y[0])):
				self.average_scores[num_round][i] = np.mean(self.observed_y[:,i])
			print((1.0 - num_round/num_iter) * (time.time()-start_t)/(num_round+1))
			print("ETA: ", int((num_iter-num_round) * (time.time()-start_t)/(num_round+1)), " seconds",flush=True)
			print("------------------------------------------------------------------------",flush=True)
			print("",flush=True)
			np.savetxt("data/round"+str(num_round)+"_observed_points.txt", self.observed_x)
			np.savetxt("data/round"+str(num_round)+"_observed_scores.txt", self.observed_x)
			np.savetxt("data/round"+str(num_round)+"_observed_scores_scalarized.txt", self.observed_y_scalarized)
			gc.collect()
		plt.plot(range(len(self.observed_y_scalarized)), self.observed_y_scalarized)   
		plt.show()


# In[5]:


initial_x = np.random.randn(10,3)
initial_y = np.array([multi_dimensional_score(i) for i in initial_x])


# In[6]:


bayesian_optimizer = BO(initial_x, initial_y)


# In[ ]:


bayesian_optimizer.run_BO(35)


# In[ ]:


bayesian_optimizer.average_scores


# In[ ]:
np.savetxt("average_scores.txt", bayesian_optimizer.average_scores)

for i in range(len(bayesian_optimizer.average_scores[0])):
	plt.plot(range(5), bayesian_optimizer.average_scores[:,i]/np.abs(bayesian_optimizer.average_scores[0,i]))
plt.show()


# In[ ]:




