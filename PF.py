#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 02:47:44 2021

@author: ruimin
"""

"""
Created on Mon May  3 21:46:11 2021

@author: RuiMIN

This class implements a Particle filter.

1. Prior distribution:

Initialization:
    the initial particles: x_{0}ˆ{(i)} ∼ p(x_{0})
    the initial weights:   w_{0}ˆ{(i)} = 1/Np
    
Sequential Processing:
    Prediction step:
        the particles: the x_{t}ˆ{(i)} ∼ p(x_{t}|x_{t}ˆ{(i)})
    Correction step:
        the weights:   w_{t}ˆ{(i)} = w_{t-1}ˆ{(i)}p(y_{t}|x_{t}ˆ{(i)})
        normalize the weights
    Estimation step:
         p(X_{t}|Y_{1:t})\approx \sum_{i=1}^{N_{p}} w_{t}^{(i)} \delta_{X}_{t}^{(i)}}
    Resampling step:
        if the N_{eff} < a threshold:
            resamling
"""


import numpy as np
from numpy import dot, zeros, eye, ones, sqrt, exp, pi
from numpy.random import randn, multivariate_normal, random
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

class ParticleFilter:
    
    def __init__(self, H, Q, R, dt, F, x0, P0, Np):
        
        # Model parameters
        self.H = H    # Measurement function
        self.dim_y, self.dim_x = np.shape(self.H) # Dimensions of model
        
        # Noise
        self.Q = Q               # Process noise covariance matrix
        self.R = R               # Measurement noise covariance matrix 
        
        # Lorenz96 model parameters
        self.dt = dt            # Time period
        self.F = F              # Force
        
        self.Np = Np     # Number of particles
        
        # Initialization
        self.particles = multivariate_normal(x0, P0, self.Np).T   # Initial particles (dim_x, Np)
        self.weights = 1 / self.Np * ones((self.Np,1))       # initial weights (Np,1)
        
    
    def prediction(self):
        
        for i in range(self.Np):
            CurrentMean = rungekutta4(self.particles[:, i], self.dt, self.F)
            self.particles[:,i] = CurrentMean + multivariate_normal(zeros(self.dim_x), self.Q)
            

    def correction(self, y):
        self.log_weights = zeros((self.Np, 1))
        
        for i in range(self.Np):
            # weight
            self.log_weights[i, 0] = np.log(self.weights[i, 0]) + logmultivariate_normal(y, dot(self.H, self.particles[:, i]), self.R)
   
        self.log_weights = self.log_weights - np.max(self.log_weights)  
        self.weights = exp(self.log_weights)
        #  Normalization of weights
        self.weights /= sum(self.weights)
        
    def estimate(self):
        self.x_est = dot(self.particles, self.weights)
        return self.x_est
    
        
        
    def resampling(self):
        index = multinomial_resample(self.weights)
        self.particles = self.particles[:, index]
        self.weights = 1 / self.Np * ones((self.Np,1))