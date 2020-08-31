#!/usr/bin/env python
# coding: utf-8

# In[49]:


from __future__ import division 
import sys
import numpy as np
import csv
import pandas as pd
from numba import jit
from Bio.Seq import Seq
from Bio import SeqIO
import scipy.io as sio
import matplotlib.pyplot as plt


# In[50]:


def get_seq(filename, get_header = False):
	#assert filename.endswith('.fasta'), 'Not a fasta file.'
	
	records = list(SeqIO.parse(filename, "fasta"))
	records_seq = [i.seq for i in records]
	headers = [i.description for i in records]
	if get_header == True:
		return records_seq, headers
	else:
		return records_seq

def index_J(length, j,k,a1,a2):
	triangle = np.arange(length-1 ,0,-1)
	return np.int(np.sum(triangle[:j]*21*21) + (k-j-1)*21*21 + 21*(a1) + a2)

def compute_E(msa):
	length = n
	energy_list = []
	for i in range(len(msa)):
		seq = msa[i]
		energy = 0
		for j in range(length):
			energy += Htensor[j,int(seq[j])]
			for k in range(j+1,length):
				index = index_J(length,j,k,seq[j],seq[k])
				energy += J_array[index]
		energy_list.append(energy)
	return energy_list


# In[55]:

dca_para = pd.read_csv('parameters_1600.txt', sep = ' ',header=None)
dict_new = {'-ACDEFGHIKLMNPQRSTVWY'[i]:i for i in range(21)}


# In[59]:

new = get_seq('test.fasta')
new = [i[:16]+i[18:44]+i[45:] for i in new]
n = len(new[0])
Htensor = np.array(dca_para[-21*n:][3]).reshape([n,21])
Jtensor = dca_para[:-21*n][[1,2,3,4,5]]
J_array = np.array(Jtensor[5])

print(new)
# In[60]:


new_bmint = []
for i in new:
	new_bmint.append(np.array([dict_new[j] for j in i]))
dcaE = compute_E(np.array(new_bmint))
print(dcaE)
class DCA_scorer():			
	def get_dca_scores_from_seq(self,sequence):
		new = [sequence]
		new = [i[:16]+i[18:44]+i[45:] for i in new]
		n = len(new[0])
		Htensor = np.array(dca_para[-21*n:][3]).reshape([n,21])
		Jtensor = dca_para[:-21*n][[1,2,3,4,5]]
		J_array = np.array(Jtensor[5])

		new_bmint = []
		for i in new:
			new_bmint.append(np.array([dict_new[j] for j in i]))
		dcaE = compute_E(np.array(new_bmint))
		return dcaE

# In[61]:




# In[ ]:





# In[ ]:




