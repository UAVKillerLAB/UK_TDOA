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

Q = np.mat([[5, 0], [0, 10]])  # 过程噪声均值，较小
R = np.mat([[100, 0], [0, 100]])  # 观测噪声均值
A = np.mat([[1, T], [0, 1]])  # 状态转移矩阵
H = np.mat([[1, 0], [0, 1]])  # 观测矩阵
random2 = random.randint(2, N, size=(2, N))  # 生成2xN的随机整数矩阵
W = np.sqrt(Q)*random2                    # 对矩阵进行求方根使用numpy库
V = np.sqrt(R)*random2
for i in range(1, N+1, 1):

    X[:, i:i+1] = A*X[:, i-1:i]+W[:, i:i+1]  # 状态方程，目标轨迹
    Z[:, i:i+1] = H*X[:, i:i+1]+V[:, i:i+1]  # 观测方程，目标观测

Xkf = np.zeros((2, N))
Xkf[:, 1] = X[:, 1]
P0 = eye(2)
# 算法更新
for i in range(1, N+1, 1):
    Xn = A*Xkf[:, i-1:i]
    P1 = A*P0*A.T+Q
    K = P1*H.T*inv(H*P1*H.T+R)
    Xkf[:, i:i+1] = Xn+K*(Z[:, i:i+1]-H*Xn)
    P0 = (eye(2)-K*H)*P1

plt.figure()
plt.plot(Xkf, Xkf, 'ro', markersize=0.1)
plt.plot(X, X, 'bo', markersize=1)
print(Xkf)
plt.show()