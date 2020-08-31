#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import division 
import sys
import numpy as np
import pickle
from numba import jit
import time
from ipywidgets import interact, interactive, fixed, interact_manual

from scipy.spatial import distance
from scipy.stats import scoreatpercentile 

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.animation as animation

import torch
import torch.nn as nn
import torch.utils.data
from sklearn.decomposition import IncrementalPCA

sys.path.append('./source')
import toolkit
sys.path.append('../')

device = torch.device("cpu")


# ---
# ## Import the dataset and VAE model
# 
# ### Import dataset
# We are going to use the following annotations in this notebook:
# * **$N$** number of samples
# * **$q$** number of one-hot features
# * **$n$** number of amino acid residues in a sequence

# In[43]:


# The -n argument of ./preprocessing.py
proteinname = 'SH3'
path = 'Outputs/'

# Import the Potts sequence. 
parameters = pickle.load(open(path + proteinname + ".db", 'rb'))
index = parameters['index']
q_n = parameters['q_n'] # Number of possible residues on each position


# In[44]:


alignment = 'Outputs/Final_New_Proteins_tosubmit.fasta'
sequence = [str(i) for i in toolkit.get_seq(alignment)]
sequence = [i[:16]+i[18:44]+i[45:] for i in toolkit.get_seq(alignment)]



# In[45]:


msa = 'Inputs/sh3_59.fasta'
msa = [str(i) for i in toolkit.get_seq(msa)]


# In[46]:


v_traj_onehot, _ = toolkit.convert_potts(sequence, index)

#v_traj_onehot, _ = toolkit.convert_potts(msa, index)


# In[47]:


N=np.size(v_traj_onehot,axis=0) #number of samples 
q=np.size(v_traj_onehot,axis=1) #number of one-hot features
n=np.size(q_n) # number of amino acid residues in a sequences


# ### Import VAE

# In[48]:


class VAE(nn.Module):
	def __init__(self, q, d):
		super(VAE, self).__init__()
		self.hsize=int(1.5*q) # size of hidden layer
		
		self.en1 = nn.Linear(q, self.hsize)
		self.en2 = nn.Linear(self.hsize, self.hsize) #
		self.en3 = nn.Linear(self.hsize, self.hsize)
		self.en_mu = nn.Linear(self.hsize, d)
		self.en_std = nn.Linear(self.hsize, d) # Is it logvar?
		
		self.de1 = nn.Linear(d, self.hsize)
		self.de2 = nn.Linear(self.hsize, self.hsize) #
		self.de22 = nn.Linear(self.hsize, self.hsize)
		self.de3 = nn.Linear(self.hsize, q)     
 
		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()
		self.tanh = nn.Tanh()        
		self.softmax = nn.Softmax(dim=1)
		
		self.dropout1 = nn.Dropout(p=0.3)
		self.dropout2 = nn.Dropout(p=0.3)
		
		self.bn1 = nn.BatchNorm1d(self.hsize) # batchnorm layer
		self.bn2 = nn.BatchNorm1d(self.hsize)
		self.bn3 = nn.BatchNorm1d(self.hsize)
		self.bnfinal = nn.BatchNorm1d(q)  

		#replace tanh with relu
	def encode(self, x):
		"""Encode a batch of samples, and return posterior parameters for each point."""
		x = self.tanh(self.en1(x)) # first encode
		x = self.dropout1(x) 
		x = self.tanh(self.en2(x))
		x = self.bn1(x)
		x = self.tanh(self.en3(x)) # second encode
		return self.en_mu(x), self.en_std(x) # third (final) encode, return mean and variance
	
	def decode(self, z):
		"""Decode a batch of latent variables"""
		z = self.tanh(self.de1(z))
		z = self.bn2(z)
		z = self.tanh(self.de2(z))
		z = self.dropout2(z)
		z = self.tanh(self.de22(z))
		
		# residue-based softmax
		# - activations for each residue in each position ARE constrained 0-1 and ARE normalized (i.e., sum_q p_q = 1)
		z = self.bn3(z)
		z = self.de3(z)
		z = self.bnfinal(z)
		z_normed = torch.FloatTensor() # empty tensor?
		z_normed = z_normed.to(device) # store this tensor in GPU/CPU
		for j in range(n):
			start = np.sum(q_n[:j])
			end = np.sum(q_n[:j+1])
			z_normed_j = self.softmax(z[:,start:end])
			z_normed = torch.cat((z_normed,z_normed_j),1)
		return z_normed
	
	def reparam(self, mu, logvar): 
		"""Reparameterisation trick to sample z values. 
		This is stochastic during training, and returns the mode during evaluation.
		Reparameterisation solves the problem of random sampling is not continuous, which is necessary for gradient descent
		"""
		if self.training:
			std = logvar.mul(0.5).exp_() 
			eps = std.data.new(std.size()).normal_() # normal distribution
			return eps.mul(std).add_(mu)
		else:
			return mu      
	
	def forward(self, x):
		"""Takes a batch of samples, encodes them, and then decodes them again to compare."""
		mu, logvar = self.encode(x.view(-1, q)) # get mean and variance
		z = self.reparam(mu, logvar) # sampling latent variable z from mu and logvar
		return self.decode(z), mu, logvar
	
	def loss(self, reconstruction, x, mu, logvar): 
		"""ELBO assuming entries of x are binary variables, with closed form KLD."""
		bce = torch.nn.functional.binary_cross_entropy(reconstruction, x.view(-1, q))
		KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
		# Normalise by same number of elements as in reconstruction
		KLD /= x.view(-1, q).data.shape[0] * q 
		return bce + KLD
	
	def get_z(self, x):
		"""Encode a batch of data points, x, into their z representations."""
		mu, logvar = self.encode(x.view(-1, q))
		return self.reparam(mu, logvar)


# In[49]:


# Load the trained VAE
d=3
model = VAE(q,d)
model_init = VAE(q,d)
model.load_state_dict(torch.load('source/VAE.pyt',map_location='cpu'))
model.eval()


class VAE_Operator():
		def get_score_from_z(self, z):
			z= [z]
			parameters = pickle.load(open("Outputs/SH3.db", 'rb'))
			q_n = parameters['q_n']
			index = parameters['index']
			v_traj_onehot = parameters['onehot']

			N=np.size(v_traj_onehot,axis=0)
			q=np.size(v_traj_onehot,axis=1)
			n=np.size(q_n)

			data = torch.FloatTensor(z).to(device)
			data = model.decode(data)
			v_gen = data.cpu().detach().numpy()
			
			v_traj_onehot = v_gen
			pred_ref,_,_ = model(torch.FloatTensor(v_traj_onehot))
			p_weight = pred_ref.cpu().detach().numpy()
			log_p_list = np.array(toolkit.make_logP(v_traj_onehot, p_weight,q_n))
			return log_p_list
		def get_seq_from_z(self, z):
			parameters = pickle.load(open("Outputs/SH3.db", 'rb'))
			q_n = parameters['q_n']
			index = parameters['index']
			v_traj_onehot = parameters['onehot']

			N=np.size(v_traj_onehot,axis=0)
			q=np.size(v_traj_onehot,axis=1)
			n=np.size(q_n)
			z_gen=[z]

			data = torch.FloatTensor(z_gen).to(device)
			data = model.decode(data)
			v_gen = data.cpu().detach().numpy()
			sample_list = []

			for i in range(len(z_gen)): # number of sampling points
				v_samp_nothot = toolkit.sample_seq(0, q, n, q_n, i, v_gen)
				sample_list.append(v_samp_nothot)
				sequences = toolkit.convert_alphabet(np.array(sample_list), index, q_n)

			sequence = sequences[0][:16]+"DD"+sequences[0][16:]
			sequence = sequence[:44] + "A" + sequence[44:]
			return(sequence)

