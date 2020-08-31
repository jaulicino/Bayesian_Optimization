#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

# In[2]:


# The -n argument of ./preprocessing.py
proteinname = 'SH3'
path = 'Outputs/'

# Import the Potts sequence. 
parameters = pickle.load(open(path + proteinname + ".db", 'rb'))
index = parameters['index']
q_n = parameters['q_n'] # Number of possible residues on each position


# In[3]:


alignment = 'Outputs/Final_New_Proteins_tosubmit.fasta'
sequence = [str(i) for i in toolkit.get_seq(alignment)]
sequence = [i[:16]+i[18:44]+i[45:] for i in toolkit.get_seq(alignment)]


# In[4]:


msa = 'Inputs/sh3_59.fasta'
msa = [str(i) for i in toolkit.get_seq(msa)]


# In[1]:


v_traj_onehot, _ = toolkit.convert_potts(sequence, index)
print(len(v_traj_onehot[:,1]))

for i in range(len(sequence)):
    print(sequence[i])
    print(len(sequence))
#v_traj_onehot, _ = toolkit.convert_potts(msa, index)


# In[6]:


N=np.size(v_traj_onehot,axis=0) #number of samples 
q=np.size(v_traj_onehot,axis=1) #number of one-hot features
n=np.size(q_n) # number of amino acid residues in a sequences


# ### Import VAE

# In[7]:


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


# In[8]:


# Load the trained VAE
d=3
model = VAE(q,d)
model_init = VAE(q,d)
model.load_state_dict(torch.load('source/VAE.pyt',map_location='cpu'))
model.eval()


# ---
# ## Model Validation
# 
# ### Internal validation: how well the model reconstruct natural sequences
# **1. Reconstruction accuracy**

# In[9]:


# Reconstruct ont-hot representation for input sequences by the trained VAE by encoding and decoding.
def reconstruct(model, sequence_list, q_n):
    model.eval()
    real_ref = torch.FloatTensor(sequence_list) 
    pred_ref, mu_ref, logvar_ref = model(real_ref)
    pred_ref = pred_ref.cpu().detach().numpy().reshape([-1,sum(q_n)])
    
    sequence_list = sequence_list.reshape([-1,sum(q_n)])
    length = np.size(sequence_list,axis=0)
    reconstruct_nothot = np.zeros([length,len(q_n)])
    
    for i in range(length):
        for j in range(len(q_n)):
            start = np.sum(q_n[:j])
            end = np.sum(q_n[:j+1])
            reconstruct_nothot[i,j] = np.argmax(pred_ref[i,start:end])
    return (reconstruct_nothot)


# In[10]:


real_nohot_list = toolkit.convert_nohot(v_traj_onehot, q_n)
z_test = model.get_z(torch.FloatTensor(v_traj_onehot)).cpu().detach().numpy() 
test_recons = reconstruct(model, v_traj_onehot, q_n)
Hamming_list = [int(n*distance.hamming(test_recons[i],real_nohot_list[i])) for i in range(len(v_traj_onehot))]

mean_acc = np.mean((test_recons==real_nohot_list).astype(int),axis=0) # List of mean reconstruction accuracy

print("Mean accuracy = %2f" %np.mean(mean_acc))


# Explanation of the variables:  
#   
# * **real_nohot_list** int representation of MSA  
# * **z_test** The latent variables  
# * **test recons** int representation of reconstructed sequences  
# * **Hamming_list** Hamming distances from the original MSA and reconstructed sequences.   
# * **mean_acc** List of mean reconstruction accuracy at each position ($accuracy^i$), namely  
#   
# $accuracy^i=\frac{1}{N}\sum_{i=1}^{N} \delta(a_i- \hat{a}_i)$ 
#   
# Then we plot the histogram of reconstruction Hamming distances for MSA and bar plot of $accuracy^i$:

# **2. VAE log probability**
#   
# Using the model trained by MSA, the probability of each input sequence can be estimated as 
#   
# $logP(x|z) \propto log[tr(H^TP)]$  
# 
# where  
# * $H$ is an $21×L$ matrix representing the one-hot encoding of a sequence  
# * $P$ is the probability weight matrix generated by feeding the network a sequence. 
# 
# For the first step, we compute **log_p_list**, which is the list of $log[tr(H^TP)]$ for each sequence. 
# 
# *Ref. https://arxiv.org/abs/1712.03346*

# In[11]:


st_time = time.time()

pred_ref,_,_ = model(torch.FloatTensor(v_traj_onehot))
p_weight = pred_ref.cpu().detach().numpy()
log_p_list = np.array(toolkit.make_logP(v_traj_onehot, p_weight,q_n))
    
end_time = time.time()
print("Elapsed time %.2f (s)" % (end_time - st_time))


# Then we show the $H$ and $P$ matrix of a sequence of interest.  
#   
# **Hint**: Run this block, text the index of your interested sequence in the text box on the right side of the scroll bar, then press Enter.

# In[13]:


def show_matrices(test_seq):
    print("Reconstruction Hamming distance: ", Hamming_list[test_seq])
    cmap = 'hot'
    plt.rcParams.update({'font.size': 14})

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 5))
    gened = axes[0].imshow(toolkit.make_matrix(v_traj_onehot[test_seq], n, q_n),cmap = cmap)
    axes[0].set_title('one-hot encoding matirx $H$')

    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes("right", size="1%", pad=0.05)
    plt.colorbar(gened, cax=cax)

    msa = axes[1].imshow(toolkit.make_matrix(p_weight[test_seq], n, q_n),cmap = cmap)
    axes[1].set_title('Reconstruction probability weight matrix $P$')

    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="1%", pad=0.05)
    plt.colorbar(gened, cax=cax)

    plt.tight_layout()
    
interact(show_matrices, test_seq = N-1)
plt.show()


# In[ ]:



