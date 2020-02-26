import numpy as np
import matplotlib.pyplot as plt
import math
import random
from numpy import random
from numpy import *
from numpy.linalg import*
T = 1  # 雷达扫描周期
N = 80  # 总的采样次数
X = np.zeros((2, N))  # 目标真实位置
X[:, 1] = [-100, 200]  # 目标初始位置，初始坐标为（-100，200）
Z = np.zeros((2, N))  # 传感器对位置的观测
Z[:, 1] = [-100, 200]  # 观测初始化

Q = np.diag((0.5,1))  # 过程噪声均值，较小
R = 100*eye(2)  # 观测噪声均值
A = np.mat[[1,T],[0,1]]  # 状态转移矩阵
H = np.mat[[1,0],[0,1]]  # 观测矩阵
random2 = random.randint(2, N, size=(2, N))
W = np.sqrt(Q)*random2

for i in range(1,N+1,1):

    X[:, i:i+1] = A*X[:, i-1:i]+W[:, i:i+1]  # 状态方程，目标轨迹
    Z[:, i:i+1] = H*X[:, i:i+1]+W[:, i:i+1]  # 观测方程，目标观测

Xkf = zeros(2, N)
Xkf[:, 1] = X[:, 1]
P0 = eye(2)

for i in range(2,N):
    Xn = A*Xkf[:, i-1]
    P1 = A*P0*A.T+Q
    K = P1*H.T*inv(H*P1*H.T+R)
    Xkf[:, i] = Xn+K*(Z[:, i]-H*Xn)
    P0 = (eye(2)-K*H)*P1

plt.figure()
plt.plot(Xkf, Xkf, 'ro', markersize=0.1)
plt.plot(X, X, 'bo', markersize=1)
print(Xkf)