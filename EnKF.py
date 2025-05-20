# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 20:31:50 2025

@author: rmin
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


class EnsembleKalmanFilter:
    
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
        
        self.Ne = Np     # Number of ensembles
        
        # Initialization
        self.ensembles = multivariate_normal(x0, P0, self.Ne).T# Initial particles (dim_x, Np)
        
    
    def prediction(self):
       
       for i in range(self.Ne):
           self.ensembles[:,i] = rungekutta4(self.ensembles[:, i], self.dt, self.F)
           
           
       self.x = np.mean(self.ensembles, axis = 1)
       self.P = np.cov(self.ensembles)
       
    def correction(self, y):
        
        # Ne perturbed observations
        Y = zeros((self.dim_y, self.Ne))
        U = zeros((self.dim_y, self.Ne))
        for i in range(self.Ne):
            U[:,i] = multivariate_normal(zeros(self.dim_y), self.R)
            Y[:,i] = y + U[:,i]
            
        # Covariance
        self.Ru = np.cov(U)
        
        # Project system uncertainty into measurement space S = HPH' + R
        self.S = dot(self.H, dot(self.P, self.H.T)) + self.Ru
        
        # Kalman gain Ku = PH'inv(S）
        self.Ku = dot(dot(self.P, self.H.T), inv(self.S))
        
        for i in range(self.Ne):
            
            self.z =  Y[:,i] - dot(self.H, self.ensembles[:,i])
            self.ensembles[:,i] = self.ensembles[:,i] + dot(self.Ku, self.z)
        
        self.x = np.mean(self.ensembles, axis = 1)
        self.P = np.cov(self.ensembles)    
           
        
        
        
        
       