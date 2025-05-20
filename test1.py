# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 20:31:50 2025

@author: rmin
"""

import numpy as np
from numpy import dot, zeros, eye
from numpy.random import multivariate_normal, randn
from numpy.linalg import inv


def L96(x, F):
    """Lorenz 96 model with constant forcing"""
    dim = np.shape(x)[0]
    d = np.zeros(dim)
    for i in range(dim):
        d[i] = (x[(i + 1) % dim] - x[i - 2]) * x[i - 1] - x[i] + F
    return d


def rungekutta4(x, dt, F):
    q1 = L96(x, F)
    q2 = L96(x + dt * q1 / 2, F)
    q3 = L96(x + dt * q2 / 2, F)
    q4 = L96(x + dt * q3, F)
    k = dt * (q1 + 2 * q2 + 2 * q3 + q4) / 6
    x += k
    return x


class EnsembleKalmanFilter:
    def __init__(self, H, Q, R, dt, F, x0, P0, Np):
        self.H = H  # Measurement function
        self.dim_y, self.dim_x = np.shape(self.H)
        self.Q = Q  # Process noise covariance matrix
        self.R = R  # Measurement noise covariance matrix
        self.dt = dt
        self.F = F
        self.Ne = Np
        self.ensembles = multivariate_normal(x0, P0, self.Ne).T

    def prediction(self):
        for i in range(self.Ne):
            self.ensembles[:, i] = rungekutta4(self.ensembles[:, i], self.dt, self.F)
        self.x = np.mean(self.ensembles, axis=1)
        self.P = np.cov(self.ensembles)

    def correction(self, y):
        Y = zeros((self.dim_y, self.Ne))
        U = zeros((self.dim_y, self.Ne))
        for i in range(self.Ne):
            U[:, i] = multivariate_normal(zeros(self.dim_y), self.R)
            Y[:, i] = y + U[:, i]

        self.Ru = np.cov(U)
        self.S = dot(self.H, dot(self.P, self.H.T)) + self.Ru
        self.Ku = dot(dot(self.P, self.H.T), inv(self.S))

        for i in range(self.Ne):
            self.z = Y[:, i] - dot(self.H, self.ensembles[:, i])
            self.ensembles[:, i] = self.ensembles[:, i] + dot(self.Ku, self.z)

        self.x = np.mean(self.ensembles, axis=1)
        self.P = np.cov(self.ensembles)


def test_enkf():
    import matplotlib
    matplotlib.use('Agg')  # 使用无交互式后端
    import matplotlib.pyplot as plt

    dim_x = 100
    dim_y = 50
    Ne = 50
    F = 8.0
    dt = 0.01
    num_steps = 100
    Q = 0.01 * eye(dim_x)
    R = 0.1 * eye(dim_y)
    H = zeros((dim_y, dim_x))
    for i in range(dim_y):
        H[i, i] = 1

    x_true = multivariate_normal(zeros(dim_x), eye(dim_x))
    x0 = x_true + 0.5 * randn(dim_x)
    P0 = eye(dim_x)

    observations = []
    x_real_trajectory = [x_true.copy()]
    for _ in range(num_steps):
        x_true = rungekutta4(x_true, dt, F)
        x_real_trajectory.append(x_true.copy())
        y = dot(H, x_true) + multivariate_normal(zeros(dim_y), R)
        observations.append(y)

    enkf = EnsembleKalmanFilter(H, Q, R, dt, F, x0, P0, Ne)
    estimated_states = []
    for t in range(num_steps):
        enkf.prediction()
        enkf.correction(observations[t])
        estimated_states.append(enkf.x.copy())

    time_steps = range(num_steps)
    for i in range(dim_y):
        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, [x[i] for x in x_real_trajectory[:-1]], label="True State", color='blue')
        plt.plot(time_steps, [y[i] for y in observations], label="Observations", color='orange', linestyle='dashed')
        plt.plot(time_steps, [x[i] for x in estimated_states], label="Estimated State", color='green')
        plt.xlabel("Time Step")
        plt.ylabel(f"State Variable {i}")
        plt.title(f"State Variable {i}: True vs Observations vs Estimates")
        plt.legend()
        plt.savefig(f"/home/chan/Desktop/state_variable_{i}.png")  # 保存为文件
        plt.close()


# 运行测试函数
if __name__ == "__main__":
    test_enkf()