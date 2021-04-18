#!/usr/bin/env python
# coding: utf-8

# In[ ]:
"""
Contains code to run the forward-backward algorithm for inference in
Hidden Markov Models.
"""

import numpy as np
import matplotlib.pyplot as plt


def gen(pi,A,B,Ntimes):
    """
    Contains code to run the forward-backward algorithm for inference in
    Hidden Markov Models.

    """
    Q, states=generating_states(pi,A,Ntimes)
    
    O, O_mat = generating_observations(Q,B) 
    
    return Q, states, O, O_mat 
    
def generating_states(pi,A,Ntimes):

    Q=np.ones(Ntimes)*-1

    Nstates=np.size(A[:,0])

    states = np.zeros((Nstates, Ntimes))

    Q[0]=np.random.choice(Nstates,1,p=pi)[0]

    states[int(Q[0]),0]=1

    for t in range(1, Ntimes):

        Q[t]=np.random.choice(Nstates,1,p=A[int(Q[t-1])])[0]

        idx=Q[t]

        states[int(idx),t]=1

    return Q, states

def generating_observations(Q,B):

    Ntimes=np.size(Q)

    O=np.ones(Ntimes)*-1

    Nstates=np.size(B[:,0])

    Nobservations= np.size(B[0,:])

    O_mat=np.zeros((Nobservations,Ntimes))

    for t in range(Ntimes):

        O[t]=np.random.choice(Nobservations,1,p=B[int(Q[t-1])])[0]

        O_mat[int(O[t]),t]=1

    return O, O_mat

