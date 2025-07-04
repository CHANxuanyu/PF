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

import metis
from scipy import sparse



def compute_bias_var(X, X_est):
    """
    计算 bias^2 和 variance 并返回它们的全局平均值。

    参数：
      X      : ndarray (dim_x, N)        真实轨迹
      X_est  : ndarray (Ns, dim_x, N)    Ns 次估计轨迹

    返回：
      bias2  : float  全局平均 bias^2
      var     : float  全局平均 variance
    """
    Ns, dim_x, N = X_est.shape

    # 1) 先在 Ns 维上求样本均值 X_bar (dim_x, N)
    X_bar = np.mean(X_est, axis=0)

    # 2) bias^2 矩阵：(dim_x, N)
    bias2_mat = (X_bar - X)**2

    # 3) variance 矩阵：(dim_x, N)
    var_mat = np.mean((X_est - X_bar[None, :, :])**2, axis=0)

    # 4) 平均得到全局 bias², var
    bias2 = np.mean(bias2_mat)
    var   = np.mean(var_mat)

    return bias2, var

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


# def metis_partition(cov_matrix, nb_partitions):
#     """
#     使用 METIS 对协方差矩阵生成的图进行分区
#     """
#     # 计算相关性并转换为相似度权重
#     correlation = np.abs(correlation_from_covariance(cov_matrix))
#     np.fill_diagonal(correlation, 0)
#     similarity = (correlation * 1000).astype(int)  # 转换为整数权重

#     # 构建邻接表（列表的列表，每个元素是 (邻接节点, 权重)）
#     adj_list = []
#     for i in range(similarity.shape[0]):
#         neighbors = []
#         for j in range(similarity.shape[1]):
#             if i != j and similarity[i, j] > 0:
#                 neighbors.append((j, similarity[i, j]))
#         adj_list.append(neighbors)

#     # 调用 METIS 分区
#     _, parts = metis.part_graph(adj_list, nparts=nb_partitions, seed=42)
#     return np.array(parts)

def metis_partition(cov_matrix, nb_partitions):
    dim_x = cov_matrix.shape[0]
    correlation = np.abs(correlation_from_covariance(cov_matrix))
    np.fill_diagonal(correlation, 0)
    similarity = (correlation * 1000).astype(int)

    # 如果所有相似度为零，使用顺序分区
    if np.all(similarity == 0):
        # 每个块依次分配变量，确保每个块至少有一个变量
        partition = np.repeat(np.arange(nb_partitions), int(np.ceil(dim_x / nb_partitions)))[:dim_x]
        return partition.astype(int)

    # 否则构建邻接表并调用 METIS
    adj_list = []
    for i in range(dim_x):
        neighbors = []
        for j in range(dim_x):
            if i != j and similarity[i, j] > 0:
                neighbors.append((j, similarity[i, j]))
        adj_list.append(neighbors)

    _, parts = metis.part_graph(adj_list, nparts=nb_partitions, seed=42)
    return np.array(parts, dtype=int)

def safe_partition(Index, dim_x, Nb):
    """
    确保每个块至少有一个变量。
    如果某些块为空，则从非空块中随机抽取变量分配给空块。
    """
    Index = Index.copy()
    unique_blocks = np.unique(Index)
    missing_blocks = [b for b in range(Nb) if b not in unique_blocks]

    if not missing_blocks:
        return Index

    # 获取每个块中的变量索引
    block_indices = [np.where(Index == b)[0] for b in range(Nb)]
    
    # 找出非空块
    non_empty_blocks = [b for b in range(Nb) if len(block_indices[b]) > 0]

    if not non_empty_blocks:
        # 所有块都为空，使用顺序分区
        return np.repeat(np.arange(Nb), int(np.ceil(dim_x / Nb)))[:dim_x]

    # 对于每个缺失块，从非空块中抽取一个变量
    for missing_block in missing_blocks:
        # 从非空块中随机选一个块
        source_block = np.random.choice(non_empty_blocks)
        # 从该块中随机选一个变量
        idx_to_move = np.random.choice(block_indices[source_block])
        Index[idx_to_move] = missing_block

        # 更新 block_indices
        block_indices[source_block] = block_indices[source_block][block_indices[source_block] != idx_to_move]
        block_indices[missing_block] = np.append(block_indices[missing_block], idx_to_move)

    return Index

# Paramters
dt = 0.05    # Time period
F = 8        # Force
dim_x = dim_y = 40    # Dimension of the state and observation
t_end = 5            
H = eye(dim_y, dim_x)   # Observation matrix 

Q = 0.25 * eye(dim_x)
# Q = zeros((dim_x, dim_x)) # Process noise covariance matrix
R = eye(dim_y)          # Measurement noise covariance matrix

# Initial state
mean0 = F * np.ones(dim_x)    # Mean of initial state
mean0[int(dim_x / 2 - 1)] -= 0.2  # Add small perturbation to the middle variable
Cov0 = 0.01 * eye(dim_x)

N = int(t_end / dt)     # Number of time steps
Ns = 20             # Number of simulations
Np = 1000      # Number of particles

# State space model
X = zeros((dim_x, N))
Y = zeros((dim_y, N))

# # Initial state
# x = multivariate_normal(mean0, Cov0) 
# for t in range(N):
#     # State
#     x = rungekutta4(x, dt, F) + multivariate_normal(zeros(dim_x), Q)
#     X[:, t] = x
#     # Observation
#     # y_t = H * x_t + v_t ,  v_t ~ N(0, R)
#     Y[:, t] = dot(H, X[:, t]) + multivariate_normal(zeros(dim_y), R)

Nb = 16  # Number of blocks
block_size = math.ceil(dim_x / Nb)

cwd = os.getcwd()
X = pickle.load(open(cwd + '/ssm.p', "rb"))['1']
Y = pickle.load(open(cwd + '/ssm.p', "rb"))['2']

print('MPF with METIS partition')
"""
Multiple particle filter in METIS partition

"""


X_est_mpf1 = zeros(((Ns, dim_x, N)))                       # state estimate
Block_mpf1 = zeros(((Ns, dim_x, N)))


for k in range(Ns): 
    print(k)
    mpf1 = MultipleParticleFilter(H, Q, R, dt, F, mean0, Cov0, Np, Nb)
     
    for i in range(N):
        P = np.cov(mpf1.particles)
        Jm = lorenz96_jacobian(mpf1.x_est)
        P = dot(dot(Jm, P), Jm.T) + Q
        
        # 调用METIS分区
        Index = metis_partition(P, Nb)
        Index = safe_partition(Index, dim_x, Nb)
        Block_mpf1[k, :, i] = Index
        
        mpf1.prediction(Index)
        mpf1.correction(Y[:, i])
        X_est_mpf1[k, :, i] = mpf1.estimate().reshape(-1)
        mpf1.resampling()
        

mse_mpf1 = np.mean(MSE(X, X_est_mpf1)[1])




print('MPF with HC partition')
"""
Multiple particle filter in HC partition

"""


X_est_mpf2 = zeros(((Ns, dim_x, N)))                       # state estimate
Block_mpf2 = zeros(((Ns, dim_x, N)))

for k in range(Ns): 
    print(k)
    # Multiple Particle filter
    mpf2 = MultipleParticleFilter(H, Q, R, dt, F, mean0, Cov0, Np, Nb)
     
    for i in range(N):
        
        
        #blocking step
        P = np.cov(mpf2.particles)
        Jm = lorenz96_jacobian(mpf2.x_est)
        P = dot(dot(Jm, P), Jm.T) + Q
        correlation = np.abs(correlation_from_covariance(P))
        correlation[correlation < 0.3] = 0
        distance = 1 - np.abs(correlation)
        distance[distance == 1] = 10
        
        
        model = AgglomerativeClustering(n_clusters=Nb, metric='precomputed', linkage='average')
        Index = model.fit_predict(distance)
        Block_mpf2[k, :, i] =  Index
        
        mpf2.prediction(Index)
        # blocking step
        mpf2.correction(Y[:, i])
        # Estimation
        X_est_mpf2[k, :, i] = mpf2.estimate().reshape(-1)
        # Resampling
        mpf2.resampling()
        
        

mse_mpf2 = np.mean(MSE(X, X_est_mpf2)[1])

# Save the datas
cwd = os.getcwd()

dict_SSM = {
        '1':  X,
        '2':  Y
        }

dict_mpf2 = {
        '1':  X_est_mpf2,
        '2':  Block_mpf2,
        '3':  mse_mpf2
        }



dict_mpf1 = {
        '1':  X_est_mpf1,
        '2':  Block_mpf1,
        '3':  mse_mpf1
        }

pickle.dump(dict_SSM, open(cwd+'/ssm.p', 'wb'))
pickle.dump(dict_mpf2, open(cwd+'/RandomTime_Np400_Nb10.p', 'wb'))
pickle.dump(dict_mpf1, open(cwd+'/HC_Np400_Nb10.p', 'wb'))


# 加载保存的文件
dict_mpf2 = pickle.load(open(cwd+'/RandomTime_Np400_Nb10.p', 'rb'))
dict_mpf1 = pickle.load(open(cwd+'/HC_Np400_Nb10.p', 'rb'))

# 打印 MSE
print('MSE for MPF with HC partition:', dict_mpf2['3'])
print('MSE for MPF with METIS partition:', dict_mpf1['3'])

bias2_mpf1, var_mpf1 = compute_bias_var(X, X_est_mpf1)
bias2_mpf2, var_mpf2 = compute_bias_var(X, X_est_mpf2)

print("METIS 分区: bias^2 = {:.6f}, variance = {:.6f}".format(bias2_mpf1, var_mpf1))
print("HC 分区: bias^2 = {:.6f}, variance = {:.6f}".format(bias2_mpf2, var_mpf2))