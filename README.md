# Codes for Bayesian Optimization
*Written by Joe Aulicino during summer 2020 for use in Dr. Andrew Ferguson's Lab of the Uchicago PME*

The main functionality is contained in the *main.ipynb* ipython notebook. Currently, a mock functionality score is  calculated from a sequence's Hamming distance to a equal length sequence of alanine. 

##Details##
The BO object in *main.ipynb* contains the main data and methods for operation and analysis of Bayesian Optimization. The data is fitted using a *GaussianProcessRegressor()* from *sklearn*. 

##Bayesian Optimization##
Each round of Bayesian Optimization, a the module randomly samples (1,000,000 by default) random sequences. The sequence which maximizes the acquisition function (EI by default) is chosen as the sequence to measure next round


