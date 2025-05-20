#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 16:50:57 2021

@author: ruimin
"""

from PF import ParticleFilter
from BPF import BlockParticleFilter
from MPF import MultipleParticleFilter
from EnKF import EnsembleKalmanFilter
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
import math
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import hdbscan
from numpy import zeros, eye, dot
from numpy.random import multivariate_normal
import copy
from copy import deepcopy
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import pairwise_distances_argmin_min


# 自定义平衡层次聚类
class BalancedAgglomerativeClustering:
    def __init__(self, n_clusters, min_size, max_size):
        self.n_clusters = n_clusters
        self.min_size = min_size
        self.max_size = max_size

    def fit(self, distance_matrix):
        n_samples = distance_matrix.shape[0]
        clusters = [{i} for i in range(n_samples)]
        
        while len(clusters) > self.n_clusters:
            # 找到最短距离的两个簇
            min_distance = float('inf')
            merge_candidates = (-1, -1)
            
            for i, cluster_i in enumerate(clusters):
                for j, cluster_j in enumerate(clusters):
                    if i >= j:
                        continue
                    # 合并后簇的大小
                    merged_size = len(cluster_i) + len(cluster_j)
                    if merged_size > self.max_size:
                        continue
                    # 计算簇间距离
                    distance = np.mean([distance_matrix[a][b] for a in cluster_i for b in cluster_j])
                    if distance < min_distance:
                        min_distance = distance
                        merge_candidates = (i, j)
            
            # 合并最近的簇
            if merge_candidates != (-1, -1):
                i, j = merge_candidates
                clusters[i].update(clusters[j])
                clusters.pop(j)
            else:
                break  # 无法再合并
            
        # 分配标签
        labels = np.zeros(n_samples, dtype=int)
        for label, cluster in enumerate(clusters):
            for index in cluster:
                labels[index] = label
        self.labels_ = labels
        return self

def MSE(X, X_est):
   
    Ns = np.shape(X_est)[0]  # number of simulation
    dim_x = np.shape(X_est)[1]  # dimensions of state
    N = np.shape(X_est)[2]  # time steps
    se = np.zeros(((Ns, dim_x, N)))
    mse = np.zeros(Ns)
    for k in range(Ns):
        se[k, :, :] = (X - X_est[k, :, :]) ** 2
        mse[k] = np.mean(se[k, :, :])
    
    return se, mse


def lorenz96_jacobian(x):
    N = len(x)
    J = np.zeros((N, N))
    for i in range(N):
        J[i, i] = -1
        J[i, (i - 1) % N] = x[(i + 1) % N] - x[(i - 2) % N]
        J[i, (i + 1) % N] = x[(i - 1) % N]
        J[i, (i - 2) % N] = -x[(i - 1) % N]
    return J


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

Q = 0.16 * eye(dim_x)
# Q = zeros((dim_x, dim_x)) # Process noise covariance matrix
R = eye(dim_y)          # Measurement noise covariance matrix

# Initial state
mean0 = F * np.ones(dim_x)    # Mean of initial state
mean0[int(dim_x / 2 - 1)] -= 0.2  # Add small perturbation to the middle variable
Cov0 = 0.01 * eye(dim_x)


N = int(t_end / dt)     # Number of time steps
Ns = 20             # Number of simulations
Np = 200      # Number of particles



# State space model
X = zeros((dim_x, N))
Y = zeros((dim_y, N))


# Initial state
# x = multivariate_normal(mean0, Cov0) 
# for t in range(N):
#     # State
#     x = rungekutta4(x, dt, F) + multivariate_normal(zeros(dim_x), Q)
#     X[:, t] = x
#     # Observation
#     # y_t = H * x_t + v_t ,  v_t ~ N(0, R)
#     Y[:, t] = dot(H, X[:, t]) + multivariate_normal(zeros(dim_y), R)
    
# State space model
cwd = os.getcwd()
X = pickle.load(open(cwd + '/ssm.p', "rb"))['1']
Y = pickle.load(open(cwd + '/ssm.p', "rb"))['2']

print("X 的前5个元素:", X[:5] if isinstance(X, (list, np.ndarray)) else "无法直接切片")
print("Y 的前5个元素:", Y[:5] if isinstance(Y, (list, np.ndarray)) else "无法直接切片")
# print('Ensemble Kalman Filter')
# """
# Ensemble Kalman Filter 
# """

# X_est_Enkf = zeros(((Ns, dim_x, N)))   


# for k in range(Ns):
#     print(k)
#     # Block Kalman Filter 
#     Enkf = EnsembleKalmanFilter(H, Q, R, dt, F, mean0, Cov0, Np)
  
#     for i in range(N):
#         # Prediction step 
#         Enkf.prediction()
#         # Update step
#         Enkf.correction(Y[:, i])
#         X_est_Enkf[k, :, i] = Enkf.x.reshape(-1)
        
# mse_Enkf = np.mean(MSE(X, X_est_Enkf)[1])


# print('PF')
# """
# Particle filter in prior distribution case
# """


# X_est_pf = zeros(((Ns, dim_x, N)))                       # state estimate    

# for k in range(Ns):
#     print(k)
#     # Particle filter
#     pf = ParticleFilter(H, Q, R, dt, F, mean0, Cov0, Np)
   
#     for i in range(N):
        
#         pf.prediction()
        
#         pf.correction(Y[:, i])
       
#         X_est_pf[k, :, i] = pf.estimate().reshape(-1)

#         pf.resampling()

# mse_pf = np.mean(MSE(X, X_est_pf)[1])


# Nb = 10
# block_size = math.ceil(dim_x / Nb)

# # Partioning
# Index_block = zeros((block_size, dim_x))
# l = sorted([int(i) for i in range(Nb)] * block_size)
# for i in range(block_size):
#     Index_block[i, :] = l[i:] + l[:i]
# Index_block = Index_block.astype(int)
       


# print('BPF with naive partition')
# """
# Block particle filter in 1st partition

# """


# X_est_bpf = zeros(((Ns, dim_x, N)))                       # state estimate


# for k in range(Ns): 
#     print(k)
#     # Block Particle filter
#     bpf = BlockParticleFilter(H, Q, R, dt, F, mean0, Cov0, Np, Nb)
    
#     for i in range(N):
#         bpf.prediction()
        
        
#         bpf.correction(Y[:, i], Index_block[0, :])
#         # Estimation
#         X_est_bpf[k, :, i] = bpf.estimate().reshape(-1)
#         # Resampling
#         bpf.resampling()
        
# mse_bpf = np.mean(MSE(X, X_est_bpf)[1])

# print('MPF with naive partition')
# """
# Multiple particle filter in 1st partition

# """


# X_est_mpf = zeros(((Ns, dim_x, N)))                       # state estimate


# for k in range(Ns): 
#     print(k)
#     # Multiple Particle filter
#     mpf = MultipleParticleFilter(H, Q, R, dt, F, mean0, Cov0, Np, Nb)
    
#     for i in range(N):
#         mpf.prediction(Index_block[0, :])
        
        
#         mpf.correction(Y[:, i])
#         # Estimation
#         X_est_mpf[k, :, i] = mpf.estimate().reshape(-1)
#         # Resampling
#         mpf.resampling()
        

# mse_mpf = np.mean(MSE(X, X_est_mpf)[1])


Nb = 10
block_size = math.ceil(dim_x / Nb)


print('MPF with random partition')
"""
Multiple particle filter in 1st partition

"""


X_est_mpf = zeros(((Ns, dim_x, N)))                       # state estimate
Block_randomTime = zeros(((Ns, dim_x, N)))

for k in range(Ns): 
    print(k)
    # Multiple Particle filter
    mpf = MultipleParticleFilter(H, Q, R, dt, F, mean0, Cov0, Np, Nb)
    
    for i in range(N):
        
        
        # random patitioning
        dim = np.random.permutation(dim_x)
        block_index = np.zeros(dim_x)
        for m in range(Nb): 
            index  = dim[np.arange(block_size) + m * block_size] 
            block_index[index] = m   
        block_index = block_index.astype(int)
        Block_randomTime[k, :, i] =  block_index 
        
        
        mpf.prediction(block_index)
        
        mpf.correction(Y[:, i])
        # Estimation
        X_est_mpf[k, :, i] = mpf.estimate().reshape(-1)
        # Resampling
        mpf.resampling()
        

mse_mpf = np.mean(MSE(X, X_est_mpf)[1])


print('MPF with proposed partition')
"""
Multiple particle filter in 1st partition

"""


X_est_mpf1 = zeros(((Ns, dim_x, N)))                       # state estimate
Block_mpf1 = zeros(((Ns, dim_x, N)))

for k in range(Ns): 
    print(k)
    # Multiple Particle filter
    mpf1 = MultipleParticleFilter(H, Q, R, dt, F, mean0, Cov0, Np, Nb)
     
    for i in range(N):
        
        
        #blocking step
        P = np.cov(mpf1.particles)
        Jm = lorenz96_jacobian(mpf1.x_est)
        P = dot(dot(Jm, P), Jm.T) + Q
        correlation = np.abs(correlation_from_covariance(P))
        correlation[correlation < 0.3] = 0
        distance = 1 - np.abs(correlation)
        distance[distance == 1] = 10
        # min_size = dim_x/Nb
        # max_size = dim_x/Nb
        # model = BalancedAgglomerativeClustering(n_clusters = Nb, min_size = min_size, max_size=max_size)
        # model.fit(distance)
        # Index = model.labels_
        
        
        
        # model = AgglomerativeClustering(n_clusters=Nb, metric='precomputed', linkage='average')
        # Index = model.fit_predict(distance)
        # Block_mpf1[k, :, i] =  Index

        # model = hdbscan.HDBSCAN(
        #     metric='precomputed',
        #     min_cluster_size=4,
        #     min_samples=1,
        #     cluster_selection_method='eom'
        # )
        # model.fit(distance.astype('double'))
        # # 处理噪声和簇数
        # Index = model.labels_.copy()
        # noise_mask = (Index == -1)
        # if noise_mask.any():
        #     # 分配噪声点到最近簇
        #     valid_clusters = np.unique(Index[Index != -1])
        #     centroids = np.array([distance[Index == c].mean(axis=0) for c in valid_clusters])
        #     nearest = pairwise_distances_argmin_min(
        #         distance[noise_mask], centroids, metric='precomputed'
        #     )[0]
        #     Index[noise_mask] = valid_clusters[nearest]
        # # 强制合并到Nb簇
        # if len(np.unique(Index)) != Nb:
        #     Z = linkage(distance, method='average')
        #     Index = fcluster(Z, t=Nb, criterion='maxclust') - 1
        # Block_mpf1[k, :, i] =  Index





        model = hdbscan.HDBSCAN(
            metric='precomputed',
            min_cluster_size=4,
            min_samples=1,
            cluster_selection_method='eom'
        )
        model.fit(distance.astype('double'))

        # 处理标签
        Index = model.labels_.copy()

        # 噪声处理
        noise_mask = (Index == -1)
        if noise_mask.any():
            valid_clusters = np.unique(Index[Index != -1])
        
            # 基于距离矩阵选择簇中心
            cluster_centers = []
            for c in valid_clusters:
                cluster_mask = (Index == c)
                cluster_distances = distance[cluster_mask][:, cluster_mask]
                centrality = cluster_distances.mean(axis=1)
                center_idx = np.argmin(centrality)
                cluster_centers.append(np.where(cluster_mask)[0][center_idx])
        
            # 计算噪声点到簇中心的距离
            noise_indices = np.where(noise_mask)[0]
            dist_matrix = distance[noise_indices][:, cluster_centers]

            # 分配最近簇
            nearest_clusters = np.argmin(dist_matrix, axis=1)
            Index[noise_indices] = valid_clusters[nearest_clusters]

            # 强制合并到Nb簇
        if len(np.unique(Index)) != Nb:
            Z = linkage(distance, method='average')
            Index = fcluster(Z, t=Nb, criterion='maxclust') - 1

        Block_mpf1[k, :, i] = Index
        






        
        mpf1.prediction(Index)
        
        
        mpf1.correction(Y[:, i])
        # Estimation
        X_est_mpf1[k, :, i] = mpf1.estimate().reshape(-1)
        # Resampling
        mpf1.resampling()
        

mse_mpf1 = np.mean(MSE(X, X_est_mpf1)[1])

# Save the datas
cwd = os.getcwd()

dict_SSM = {
        '1':  X,
        '2':  Y
        }

dict_mpf_randomTime = {
        '1':  X_est_mpf,
        '2':  Block_randomTime,
        '3':  mse_mpf
        }



dict_mpf1 = {
        '1':  X_est_mpf1,
        '2':  Block_mpf1,
        '3':  mse_mpf1
        }

# pickle.dump(dict_SSM, open(cwd+'/ssm.p', 'wb'))
# pickle.dump(dict_mpf_randomTime, open(cwd+'/RandomTime_Np400_Nb10.p', 'wb'))
# pickle.dump(dict_mpf1, open(cwd+'/HC_Np400_Nb10.p', 'wb'))


# 加载保存的文件
# dict_mpf_randomTime = pickle.load(open(cwd+'/RandomTime_Np400_Nb10.p', 'rb'))
# dict_mpf1 = pickle.load(open(cwd+'/HC_Np400_Nb10.p', 'rb'))

# 打印 MSE
print('MSE for MPF with random partition:', dict_mpf_randomTime['3'])
print('MSE for MPF with proposed partition:', dict_mpf1['3'])
