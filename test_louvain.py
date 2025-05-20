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
from numpy import zeros, eye, dot
from numpy.random import multivariate_normal
import copy
from copy import deepcopy

import networkx as nx
import community  # python-louvain库
from collections import defaultdict
import random


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

def find_adjacent_cluster(G, target_comm, clusters):
    """找到与目标社区连接最强的相邻社区"""
    target_nodes = clusters[target_comm]
    edge_weights = defaultdict(float)
    
    for node in target_nodes:
        for neighbor in G.neighbors(node):
            neighbor_comm = partition[neighbor]
            if neighbor_comm != target_comm:
                edge_weights[neighbor_comm] += G[node][neighbor]['weight']
    
    if not edge_weights:
        return random.choice(list(clusters.keys()))
    return max(edge_weights, key=edge_weights.get)

def update_clusters(partition):
    """更新集群字典"""
    clusters = defaultdict(list)
    for node, comm in partition.items():
        clusters[comm].append(node)
    return clusters

def balance_clusters(labels, target_clusters, min_size, max_size, distance_matrix):
    """
    平衡簇大小的简单实现
    """
    unique, counts = np.unique(labels, return_counts=True)
    clusters = {u: np.where(labels == u)[0].tolist() for u in unique}
    
    # 分割过大的簇
    for comm in list(clusters.keys()):
        if len(clusters[comm]) > max_size:
            # 使用距离矩阵进行更智能的分割
            sub_dist = distance_matrix[np.ix_(clusters[comm], clusters[comm])]
            model = AgglomerativeClustering(n_clusters=2, metric="precomputed", linkage="average")
            split_labels = model.fit_predict(sub_dist)
            new_comm = max(clusters.keys()) + 1
            clusters[new_comm] = [clusters[comm][i] for i, lbl in enumerate(split_labels) if lbl == 1]
            clusters[comm] = [clusters[comm][i] for i, lbl in enumerate(split_labels) if lbl == 0]
    
    # 合并过小的簇
    while any(len(v) < min_size for v in clusters.values()):
        small_comms = [k for k, v in clusters.items() if len(v) < min_size]
        for comm in small_comms:
            if comm not in clusters:
                continue
            # 使用传入的距离矩阵计算最近簇
            distances = []
            for other_comm in clusters:
                if other_comm != comm:
                    dist = np.mean([distance_matrix[i][j] 
                                   for i in clusters[comm]
                                   for j in clusters[other_comm]])
                    distances.append((other_comm, dist))
            if distances:
                nearest = min(distances, key=lambda x: x[1])[0]
                clusters[nearest].extend(clusters[comm])
                del clusters[comm]
    
    # 最终强制填充空簇
    existing = set(clusters.keys())
    for m in range(target_clusters):
        if m not in existing:
            # 随机选择一个节点填充
            all_nodes = set(range(dim_x))
            used = set().union(*clusters.values())
            available = list(all_nodes - used)
            if not available:
                available = list(all_nodes)
            clusters[m] = [np.random.choice(available)]


    # 重新分配标签
    new_labels = np.zeros_like(labels)
    for new_id, comm in enumerate(clusters):
        for idx in clusters[comm]:
            new_labels[idx] = new_id
    return new_labels

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


# # Initial state
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


print('MPF with proposed partition')
"""
Multiple particle filter in 1st partition

"""


X_est_mpf1 = zeros(((Ns, dim_x, N)))                       # state estimate
Block_mpf1 = zeros(((Ns, dim_x, N)))

# for k in range(Ns): 
#     print(k)
#     # Multiple Particle filter
#     mpf1 = MultipleParticleFilter(H, Q, R, dt, F, mean0, Cov0, Np, Nb)
     
#     for i in range(N):
        
        
#         #blocking step
#         P = np.cov(mpf1.particles)
#         Jm = lorenz96_jacobian(mpf1.x_est)
#         P = dot(dot(Jm, P), Jm.T) + Q
#         correlation = np.abs(correlation_from_covariance(P))
#         correlation[correlation < 0.3] = 0
#         distance = 1 - np.abs(correlation)
#         distance[distance == 1] = 10
#         # min_size = dim_x/Nb
#         # max_size = dim_x/Nb
#         # model = BalancedAgglomerativeClustering(n_clusters = Nb, min_size = min_size, max_size=max_size)
#         # model.fit(distance)
#         # Index = model.labels_
        
        
#         model = AgglomerativeClustering(n_clusters=Nb, metric='precomputed', linkage='average')
#         Index = model.fit_predict(distance)
#         Block_mpf1[k, :, i] =  Index
        
#         mpf1.prediction(Index)
        
        
#         mpf1.correction(Y[:, i])
#         # Estimation
#         X_est_mpf1[k, :, i] = mpf1.estimate().reshape(-1)
#         # Resampling
#         mpf1.resampling()


#  louvain
# （保留所有之前的导入和函数定义，直到参数设置部分）

# 新增调试函数
def validate_partition(partition, target):
    unique = np.unique(list(partition.values()))
    assert len(unique) == target, f"分区错误：当前社区数{len(unique)} 目标{target}"
    for comm in unique:
        assert sum(1 for v in partition.values() if v == comm) >= 1, f"空社区 {comm}"

# 修改后的社区调整函数
def adjust_to_target_clusters(partition, target, G, max_retry=3):
    """带重试机制的社区调整"""
    for _ in range(max_retry):
        clusters = update_clusters(partition)
        
        # 合并阶段
        while len(clusters) > target:
            # 找到两个最小社区
            sizes = {k: len(v) for k, v in clusters.items()}
            sorted_comms = sorted(sizes, key=lambda x: sizes[x])
            candidates = sorted_comms[:2]
            
            # 寻找最佳合并对
            best_pair = None
            max_connection = -1
            for comm in candidates:
                neighbors = defaultdict(float)
                for node in clusters[comm]:
                    for neighbor in G.neighbors(node):
                        n_comm = partition[neighbor]
                        if n_comm != comm:
                            neighbors[n_comm] += G[node][neighbor]['weight']
                if neighbors:
                    strongest = max(neighbors, key=neighbors.get)
                    if neighbors[strongest] > max_connection:
                        max_connection = neighbors[strongest]
                        best_pair = (comm, strongest)
            
            # 执行合并
            if best_pair:
                src, dest = best_pair
                for node in clusters[src]:
                    partition[node] = dest
                clusters = update_clusters(partition)
            else:
                # 强制合并最小的两个
                comm1, comm2 = candidates[:2]
                for node in clusters[comm2]:
                    partition[node] = comm1
                clusters = update_clusters(partition)
            clusters = update_clusters(partition)
        
        # 分裂阶段
        while len(clusters) < target:
            # 找到最大社区
            largest = max(clusters, key=lambda k: len(clusters[k]))
            sub_nodes = clusters[largest]
            
            # 子图划分
            subG = G.subgraph(sub_nodes)
            sub_part = community.best_partition(subG, resolution=2.0)
            if len(set(sub_part.values())) > 1:
                new_base = max(partition.values()) + 1
                for node in sub_nodes:
                    partition[node] = new_base + sub_part[node]
            else:
                # 强制平分
                mid = len(sub_nodes) // 2
                new_comm = max(partition.values()) + 1
                for i, node in enumerate(sub_nodes):
                    partition[node] = new_comm if i >= mid else largest
            clusters = update_clusters(partition)
        
        # 标签规范化
        unique_comms = sorted(set(partition.values()))
        comm_map = {old: i for i, old in enumerate(unique_comms)}
        for node in partition:
            partition[node] = comm_map[partition[node]]
        
        try:
            validate_partition(partition, target)
            return partition
        except AssertionError:
            continue
    
    # 重试失败后生成安全分区
    nodes = list(partition.keys())
    np.random.shuffle(nodes)
    partition = {node: i % target for i, node in enumerate(nodes)}
    return partition

# 修改后的主循环部分
for k in range(Ns): 
    print(f"Simulation {k+1}/{Ns}")
    mpf1 = MultipleParticleFilter(H, Q, R, dt, F, mean0, Cov0, Np, Nb)
     
    for i in range(N):
        # 协方差计算
        P = np.cov(mpf1.particles)
        Jm = lorenz96_jacobian(mpf1.x_est)
        P = dot(dot(Jm, P), Jm.T) + Q
        correlation = np.abs(correlation_from_covariance(P))
        
        # 构建网络
        G = nx.Graph()
        for i in range(dim_x):
            for j in range(i+1, dim_x):
                if correlation[i,j] > 0.3:
                    G.add_edge(i, j, weight=correlation[i,j])
        
        # 初始分区
        if len(G.edges) == 0:
            partition = {n: n % Nb for n in range(dim_x)}
        else:
            partition = community.best_partition(G, resolution=1.0)
        
        # 调整分区
        adjusted_part = adjust_to_target_clusters(
            partition.copy(), 
            target=Nb,
            G=G
        )
        
        # 转换为标签数组
        Index = np.array([adjusted_part[i] for i in range(dim_x)])
        
        # 平衡处理
        try:
            Index = balance_clusters(
                Index, Nb, 
                min_size=1, max_size=5,
                distance_matrix=1 - correlation
            )
        except:
            print(f"平衡失败，使用随机分区")
            perm = np.random.permutation(dim_x)
            Index = np.zeros(dim_x, dtype=int)
            for m in range(Nb):
                start = m * block_size
                end = min((m+1)*block_size, dim_x)
                Index[perm[start:end]] = m
        
        # 确保分区有效性
        unique, counts = np.unique(Index, return_counts=True)
        if len(unique) != Nb or (counts < 1).any():
            print(f"最终验证失败，使用循环分区")
            Index = np.array([i % Nb for i in range(dim_x)])
        
        # 存储和使用分区
        Block_mpf1[k, :, i] = Index
        mpf1.prediction(Index)
        mpf1.correction(Y[:, i])
        X_est_mpf1[k, :, i] = mpf1.estimate().reshape(-1)
        mpf1.resampling()

# （保留后续的保存和输出部分）




        

mse_mpf1 = np.mean(MSE(X, X_est_mpf1)[1])



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

pickle.dump(dict_SSM, open(cwd+'/ssm.p', 'wb'))
pickle.dump(dict_mpf1, open(cwd+'/HC_Np400_Nb10.p', 'wb'))
pickle.dump(dict_mpf_randomTime, open(cwd+'/RandomTime_Np400_Nb10.p', 'wb'))



# 加载保存的文件
dict_mpf1 = pickle.load(open(cwd+'/HC_Np400_Nb10.p', 'rb'))
dict_mpf_randomTime = pickle.load(open(cwd+'/RandomTime_Np400_Nb10.p', 'rb'))


# 打印 MSE
print('MSE for MPF with proposed partition:', dict_mpf1['3'])
print('MSE for MPF with random partition:', dict_mpf_randomTime['3'])

