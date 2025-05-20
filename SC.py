#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 16:50:57 2021

@author: ruimin
"""

from pf import ParticleFilter
from partialbpf import PartialBlockParticleFilter
from sklearn.cluster import SpectralClustering
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from scipy.sparse import csgraph
from k_means_constrained import KMeansConstrained
import pickle
import os
from numpy import zeros, eye, dot
from numpy.random import multivariate_normal
import copy
from copy import deepcopy


def MSE(X, X_est):
   
    Ns = np.shape(X_est)[0]  # number of simulation
    dim_x = np.shape(X_est)[1]  # dimensions of state
    N = np.shape(X_est)[2]  # time steps
    se = np.zeros(((Ns, dim_x, N)))
    mse = np.zeros(Ns)
    for k in range(Ns):
        se[k, :, :] = (X[k, :, :] - X_est[k, :, :]) ** 2
        mse[k] = np.mean(se[k, :, :])
    
    return se, mse


def correlation_from_covariance(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation

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


# Paramters
dt = 0.05    # Time period
F = 8        # Force
dim_x = dim_y = 40    # Dimension of the state and observation
t_end = 5            
H = eye(dim_y, dim_x)   # Observation matrix 
Q = eye(dim_x) # Process noise covariance matrix
R = eye(dim_y)          # Measurement noise covariance matrix

# Initial state
mean0 = F * np.ones(dim_x)    # Mean of initial state
mean0[int(dim_x / 2 - 1)] -= 0.2  # Add small perturbation to the middle variable

# Cov0 = 5 * np.exp(-DistanceIndex / 100)     # Covariance matrix of initial state
Cov0 = 0.01 * eye(dim_x)

N = int(t_end / dt)     # Number of time steps
Ns = 200              # Number of simulations
Np = 1000       # Number of particles
# block_size = 20      # Block size
# Nb = int(dim_x / block_size)      # number of block

Nb = 9
block_size = math.ceil(dim_x / Nb)

# # Partioning
# Index_block = zeros((block_size, dim_x))
# l = sorted([int(i) for i in range(Nb)] * block_size)
# for i in range(block_size):
#     Index_block[i, :] = l[i:] + l[:i]
# Index_block = Index_block.astype(int)

# # State space model
# X = zeros(((Ns, dim_x, N)))
# Y = zeros(((Ns, dim_y, N)))

# for k in range(Ns):
#     # Initial state
#     x = multivariate_normal(mean0, Cov0)
#     for t in range(N):
#         # State
#         x = rungekutta4(x, dt, F) + multivariate_normal(zeros(dim_x), Q)
#         X[k, :, t] = x
#         # Observation
#         # y_t = H * x_t + v_t ,  v_t ~ N(0, R)
#         Y[k, :, t] = dot(H, X[k, :, t]) + multivariate_normal(zeros(dim_y), R)
       


# State space model
cwd = os.getcwd()
X = pickle.load(open(cwd + '/ssm.p', "rb"))['1']
Y = pickle.load(open(cwd + '/ssm.p', "rb"))['2']


# print('BPF with naive partition')
# """
# Block particle filter in 1st partition

# """


# X_est_bpf = zeros(((Ns, dim_x, N)))                       # state estimate
# correlation_bpf = zeros((((Ns, N, dim_x, dim_x))))

# for k in range(Ns): 
#     print(k)
#     # Block Particle filter
#     bpf = PartialBlockParticleFilter(H, Q, R, dt, F, mean0, Cov0, Np, block_size, Nb)
    
#     for i in range(N):
#         bpf.prediction()
#         covariance_bpf = np.cov(bpf.particles)
#         correlation_bpf[k, i, :, :] = np.abs(correlation_from_covariance(covariance_bpf))
        
#         bpf.correction(Y[k, :, i], Index_block[0, :])
#         # Estimation
#         X_est_bpf[k, :, i] = bpf.estimate().reshape(-1)
#         # Resampling
#         bpf.resampling()
        

# mse_bpf = np.mean(MSE(X, X_est_bpf)[1])


# print('BPF with random partition')
# """
# Block Particle filter : The partitioning is random
# That is to say, the pratitioning is always different at each time step,
# but the size of each block is the same
# """


# X_est_bpf_randomTime = zeros(((Ns, dim_x, N)))          # State estimate
# Block_randomTime = zeros(((Ns, dim_x, N)))

# for k in range(Ns): 
#     print(k)
#     # Block Particle filter
#     bpf_randomTime = PartialBlockParticleFilter(H, Q, R, dt, F, mean0, Cov0, Np, block_size, Nb)
    
#     for i in range(N):
#         # Prediction step
#         bpf_randomTime.prediction()
        
#         # random patitioning
#         dim = np.random.permutation(dim_x)
#         block_index = np.zeros(dim_x)
#         for m in range(Nb): 
#             index  = dim[np.arange(block_size) + m * block_size] 
#             block_index[index] = m   
#         block_index = block_index.astype(int)
#         Block_randomTime[k, :, i] =  block_index 
        
#         # Correction step
#         bpf_randomTime.correction(Y[k, :, i], Block_randomTime[k, :, i])
        
#         # Estimation
#         X_est_bpf_randomTime[k, :, i] = bpf_randomTime.estimate().reshape(-1)
#         # Resampling
#         bpf_randomTime.resampling()

# mse_bpf_randomTime = np.mean(MSE(X, X_est_bpf_randomTime)[1])

       

print('BPF with NSC')
"""
Block Particle filter with normalized spectral clustering (NSC)

the mean of p(X_t|Y_1:t-1) is the mean of the particles at the time step t
the covariance matrix of p(X_t|Y_1:t-1) is the covariance matrix
of the particles at the time step t

estimation of covariance matrix: np.cov

"""
X_est_bpf_NSC = zeros(((Ns, dim_x, N)))             # State estimate
correlation_bpf_NSC = np.zeros((((Ns, N, dim_x, dim_x))))
W_NSC = np.zeros((((Ns, N, dim_x, dim_x)))) # Adjacency matrix 
Block_NSC = zeros(((Ns, dim_x, N)))
eigval_NSC = zeros(((Ns, dim_x, N)))

for k in range(Ns): 
    print(k)
    # Block Particle filter
    bpf_NSC = PartialBlockParticleFilter(H, Q, R, dt, F, mean0, Cov0, Np, block_size, Nb)
    
    for i in range(N):

        # Prediction step
        bpf_NSC.prediction()
        
        # Covariance matrix estimation
        covariance_NSC = np.cov(bpf_NSC.particles)
        correlation_bpf_NSC[k, i, :, :] = np.abs(correlation_from_covariance(covariance_NSC))
        
        # Adjacency matrix
        W_NSC[k, i, :, :] = correlation_bpf_NSC[k, i, :, :]
        # Component degree
        d_NSC = np.sum(W_NSC[k, i, :, :], axis=1) 
        # D^(-1/2)
        D_NSC = np.diag(1 / np.sqrt(d_NSC))
        
        # Normalized Laplacian matrix D^{-1/2} L D^{-1/2} = I - D^{-1/2} W D^{-1/2}
        L_NSC = np.eye(dim_x) - (D_NSC.dot(W_NSC[k, i, :, :])).dot(D_NSC)
        
        # Eigenvalues and eigenvectors of normalised Laplacian matrix
        eigval_NSC[k, :, i], eigvec_NSC = np.linalg.eigh(L_NSC) 
        # K vectors corresponding k smallest eigenvalues
        Z_NSC = eigvec_NSC[:,:Nb] 
        # Normalizing the rows to norm 1
        s = np.sqrt(np.sum(Z_NSC ** 2, axis = 1))
        Z_NSC = Z_NSC / (np.tile(s, (Nb, 1)).T)
        
        # K means
        clusterizer_NSC = KMeans(n_clusters = Nb)
        clusterizer_NSC.fit(Z_NSC) 
        Block_NSC[k, :, i] = clusterizer_NSC.labels_ 
        
        #Correction step
        bpf_NSC.correction(Y[k, :, i], Block_NSC[k, :, i])
        
        # Estimation
        X_est_bpf_NSC[k, :, i] = bpf_NSC.estimate().reshape(-1)
        
        # Resampling
        bpf_NSC.resampling()
        
mse_bpf_NSC = np.mean(MSE(X, X_est_bpf_NSC)[1])


print('BPF with same size NSC')
"""
Block Particle filter with Same size normalized spectral clustering 

the mean of p(X_t|Y_1:t-1) is the mean of the particles at the time step t
the covariance matrix of p(X_t|Y_1:t-1) is the covariance matrix
of the particles at the time step t

estimation of covariance matrix: np.cov

"""
X_est_bpf_sameNSC = zeros(((Ns, dim_x, N)))             # State estimate
correlation_bpf_sameNSC = np.zeros((((Ns, N, dim_x, dim_x))))
W_sameNSC = np.zeros((((Ns, N, dim_x, dim_x)))) # Adjacency matrix 
Block_sameNSC = zeros(((Ns, dim_x, N)))
eigval_sameNSC = zeros(((Ns, dim_x, N)))


for k in range(Ns): 
    print(k)
    # Block Particle filter
    bpf_sameNSC = PartialBlockParticleFilter(H, Q, R, dt, F, mean0, Cov0, Np, block_size, Nb)
    
    for i in range(N):
        # Prediction step
        bpf_sameNSC.prediction()
        
        # Covariance matrix estimation
        covariance_sameNSC = np.cov(bpf_sameNSC.particles)
        correlation_bpf_sameNSC[k, i, :, :] = np.abs(correlation_from_covariance(covariance_sameNSC))
        
        # Adjacency matrix
        W_sameNSC[k, i, :] = correlation_bpf_sameNSC[k, i, :, :]
        # Component degree
        d_sameNSC = np.sum(W_sameNSC[k, i, :, :], axis=1) 
        # D^(-1/2)
        D_sameNSC = np.diag(1 / np.sqrt(d_sameNSC))
        
        # Normalized Laplacian matrix D^{-1/2} L D^{-1/2} = I - D^{-1/2} W D^{-1/2}
        L_sameNSC = np.eye(dim_x) - (D_sameNSC.dot(W_sameNSC[k, i, :, :])).dot(D_sameNSC)
        # Eigenvalues and eigenvectors of Laplacian matrix
        eigval_sameNSC[k, :, i], eigvec_sameNSC = np.linalg.eigh(L_sameNSC)
        # K vectors corresponding k smallest eigenvalues
        Z_sameNSC = eigvec_sameNSC[:,:Nb]
        
        # Normalizing the rows to norm 1,
        s = np.sqrt(np.sum(Z_sameNSC ** 2, axis = 1))
        Z_sameNSC = Z_sameNSC / (np.tile(s, (Nb, 1)).T)
        
        # Same size K means
        clusterizer_sameNSC = KMeansConstrained(n_clusters = Nb, size_min = 1, size_max = block_size)
        clusterizer_sameNSC.fit(Z_sameNSC)  
        Block_sameNSC[k, :, i] = clusterizer_sameNSC.labels_ 
        
        #Correction step
        bpf_sameNSC.correction(Y[k, :, i], Block_sameNSC[k, :, i])
        
        # Estimation
        X_est_bpf_sameNSC[k, :, i] = bpf_sameNSC.estimate().reshape(-1)
        
        # Resampling
        bpf_sameNSC.resampling()
        
mse_bpf_sameNSC = np.mean(MSE(X, X_est_bpf_sameNSC)[1])


print('BPF with different size NSC')
"""
Block Particle filter with Different size normalized spectral clustering 

the mean of p(X_t|Y_1:t-1) is the mean of the particles at the time step t
the covariance matrix of p(X_t|Y_1:t-1) is the covariance matrix
of the particles at the time step t

estimation of covariance matrix: np.cov

"""
X_est_bpf_diffNSC = zeros(((Ns, dim_x, N)))             # State estimate
correlation_bpf_diffNSC = np.zeros((((Ns, N, dim_x, dim_x))))
W_diffNSC = np.zeros((((Ns, N, dim_x, dim_x)))) # Adjacency matrix 
Block_diffNSC = zeros(((Ns, dim_x, N)))
eigval_diffNSC = zeros(((Ns, dim_x, N)))


for k in range(Ns): 
    print(k)
    # Block Particle filter
    bpf_diffNSC = PartialBlockParticleFilter(H, Q, R, dt, F, mean0, Cov0, Np, block_size, Nb)
    
    for i in range(N):
        # Prediction step
        bpf_diffNSC.prediction()
        
        # Covariance matrix estimation
        covariance_diffNSC = np.cov(bpf_diffNSC.particles)
        correlation_bpf_diffNSC[k, i, :, :] = np.abs(correlation_from_covariance(covariance_diffNSC))
        
        # Adjacency matrix
        W_diffNSC[k, i, :] = correlation_bpf_diffNSC[k, i, :, :]
        # Component degree
        d_diffNSC = np.sum(W_diffNSC[k, i, :, :], axis=1) 
        # D^(-1/2)
        D_diffNSC = np.diag(1 / np.sqrt(d_diffNSC))
        
        # Normalized Laplacian matrix D^{-1/2} L D^{-1/2} = I - D^{-1/2} W D^{-1/2}
        L_diffNSC = np.eye(dim_x) - (D_diffNSC.dot(W_diffNSC[k, i, :, :])).dot(D_diffNSC)
        # Eigenvalues and eigenvectors of Laplacian matrix
        eigval_diffNSC[k, :, i], eigvec_diffNSC = np.linalg.eigh(L_diffNSC)
        # K vectors corresponding k smallest eigenvalues
        Z_diffNSC = eigvec_diffNSC[:,:Nb]
        
        # Normalizing the rows to norm 1,
        s = np.sqrt(np.sum(Z_diffNSC ** 2, axis = 1))
        Z_diffNSC = Z_diffNSC / (np.tile(s, (Nb, 1)).T)
        
        # Different size K means
        clusterizer_diffNSC = KMeansConstrained(n_clusters = Nb, size_min = 1, size_max = int(1.5 * block_size))
        clusterizer_diffNSC.fit(Z_diffNSC)  
        Block_diffNSC[k, :, i] = clusterizer_diffNSC.labels_ 
        
        #Correction step
        bpf_diffNSC.correction(Y[k, :, i], Block_diffNSC[k, :, i])
        
        # Estimation
        X_est_bpf_diffNSC[k, :, i] = bpf_diffNSC.estimate().reshape(-1)
        
        # Resampling
        bpf_diffNSC.resampling()
        
mse_bpf_diffNSC = np.mean(MSE(X, X_est_bpf_diffNSC)[1])


      
# Save the datas
cwd = os.getcwd()

# dict_SSM = {
#         '1':  X,
#         '2':  Y
#         }

# dict_bpf = {
#         '1':  X_est_bpf,
#         '2':  Index_block[0, :],
#         '3':  mse_bpf
#         }


# dict_bpf_randomTime = {
#         '1':  X_est_bpf_randomTime,
#         '2':  Block_randomTime,
#         '3':  mse_bpf_randomTime
#         }



dict_NSC = {
        '1':  X_est_bpf_NSC,
        '2':  correlation_bpf_NSC,
        '3':  Block_NSC,
        '4':  mse_bpf_NSC
        }


dict_sameNSC = {
        '1':  X_est_bpf_sameNSC,
        '2':  correlation_bpf_sameNSC,
        '3':  Block_sameNSC,
        '4':  mse_bpf_sameNSC
        }


dict_diffNSC = {
        '1':  X_est_bpf_diffNSC,
        '2':  correlation_bpf_diffNSC,
        '3':  Block_diffNSC,
        '4':  mse_bpf_diffNSC
        }





# pickle.dump(dict_SSM, open(cwd+'/ssm.p', 'wb'))
# pickle.dump(dict_bpf, open(cwd+'/bpf_Np1000_Nb2.p', 'wb'))
# pickle.dump(dict_bpf_randomTime, open(cwd+'/RandomTime_Np1000_Nb2.p', 'wb'))
pickle.dump(dict_NSC, open(cwd+'/NSC_Np1000_Nb9.p', 'wb'))
pickle.dump(dict_sameNSC, open(cwd+'/sameNSC_Np1000_Nb9.p', 'wb'))
pickle.dump(dict_diffNSC, open(cwd+'/diffNSC_Np1000_Nb9.p', 'wb'))