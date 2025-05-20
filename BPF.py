#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 14:53:45 2021

@author: ruimin

BPF with partial observation variables (odd variables)

If there is at least one observation variable in this block, the block weight of
each particle is computed by the likelihood. If there is not any observation 
in this block, the weights of each particles is 1/Np.

eg: 
    
1. [X1, X2, X3, X4] are in the first block, only Y1 and Y3 are observed. So the 
weight of each particle w_t,1 ^(i) = p(Y1, Y3 / X1, X3).
2. [X1, X2, X3, X4] are in the first block, no observation. So the 
weight of each particle w_t,1 ^(i) = 1/Np.

"""



import numpy as np
from numpy import dot, zeros, ones, exp
from numpy.random import multivariate_normal, random
from numpy.linalg import inv
import copy
from copy import deepcopy

def L96(x, F):
    """Lorenz 96 model with constant forcing"""
    # Setting up vector
    dim = np.shape(x)[0]
    d = np.zeros(dim)
    # Loops over indices (with operations and Python underflow indexing handling edge cases)
    for i in range(dim):
        d[i] = (x[(i + 1) % dim] - x[i - 2]) * x[i - 1] - x[i] + F
    return d

def rungekutta4(x, dt, F):
    q1 = L96(x, F)
    q2 = L96(x + dt * q1 /2, F)
    q3 = L96(x + dt * q2 /2, F)
    q4 = L96(x + dt * q3, F)
    k =  dt * (q1 + 2 * q2 + 2 * q3 + q4) / 6
    x += k
    return x


def logmultivariate_normal(y, mean, cov):
    residual = y - mean
    logpdf = - 1 / 2 * dot(dot(residual, inv(cov)), residual) 
    # calculates the probability of x for n-dim Gaussian with mean mu and var sigma
    return logpdf

def multinomial_resample(weights):
    cumulative_sum = np.cumsum(weights)
    return np.searchsorted(cumulative_sum, random(len(weights))) 

class BlockParticleFilter:
    
    def __init__(self, H, Q, R, dt, F, x0, P0, Np, Nb):
        
        # Model parameters
        self.H = H    # Measurement function
        self.dim_y, self.dim_x = np.shape(self.H) # Dimensions of model
        
        # Noise
        self.Q = Q               # Process noise covariance matrix
        self.R = R               # Measurement noise covariance matrix 
        
        # Lorenz96 model parameters
        self.dt = dt            # Time period
        self.F = F              # Force
        
        # Parameters of block particle filter
        self.Np = Np             # Number of particles for each block
        self.Nb = Nb             # Number of block
        
        # Initialization
        self.particles = multivariate_normal(x0, P0, self.Np).T
        self.weights = 1 / self.Np * ones((self.Np, self.dim_x))
        
    
    def prediction(self):
        
        for i in range(self.Np):
            # predict
            # x = deepcopy(self.particles[:, i])
            CurrentMean = rungekutta4(self.particles[:, i], self.dt, self.F)
            self.particles[:,i] = CurrentMean + multivariate_normal(zeros(self.dim_x), self.Q)

    def correction(self, y, Index_block):
        
        self.Index_block = Index_block   # Block index of the components of state  
        self.log_weights = zeros((self.Np, self.dim_x))
        for i in range(self.Nb):
            # The index of component of the current block
            CurrentIndex = np.where(self.Index_block == i)[0]
            # print(CurrentIndex)
            # The block weights
            for j in range(self.Np):
                self.log_weights[j, CurrentIndex] = logmultivariate_normal(y[CurrentIndex], dot((self.H[CurrentIndex])[:,CurrentIndex], \
                                         self.particles[CurrentIndex, j]), (self.R[CurrentIndex])[:,CurrentIndex])
    
            self.log_weights[:, CurrentIndex] -=  np.max(self.log_weights[:, CurrentIndex[0]])
            self.weights[:, CurrentIndex] = exp(self.log_weights[:, CurrentIndex])
            #  Normalization of block weights
            self.weights[:, CurrentIndex] = self.weights[:, CurrentIndex] / np.sum(self.weights[:, CurrentIndex[0]]) 
            
            
    
    def estimate(self):
        self.x_est = np.sum(self.particles * self.weights.T, axis = 1)
        return self.x_est
                
    def resampling(self):
        
        for i in range(self.Nb):
            
            CurrentIndex = np.where(self.Index_block == i)[0]
            # print('CurrentIndex = ', CurrentIndex)
            index = multinomial_resample(self.weights[:, CurrentIndex[0]])
            self.particles[CurrentIndex,:] = (self.particles[CurrentIndex])[:, index]
            self.weights[:, CurrentIndex] = 1 / self.Np