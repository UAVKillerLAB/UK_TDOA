import numpy as np
import matplotlib.pyplot as plt
import math
import random
from numpy import random
from numpy import *
from numpy.linalg import*



#接收定位数据
D = list([
        # [99.99999302, 99.99999335],
        # [149.99998931, 149.99999005],
        # [199.99998549, 199.99998677],
        # [249.99998157, 249.99998353],
        # [299.99997757, 299.99998034],
    [0, 0],
    [65.76493554, 34.23505767],
    [100, 100],
    [65.76493532, 145.76493059],
    [-5.87629289e-15, 1.99999987e+02],
    ])
N = len(D)#计算定位数据的长度，用于算法的计算次数
S = np.matrix(D)#将其转化为矩阵，便于接下来的运算
#相关矩阵初始化
A = np.mat([[1, 0], [0, 1]])#转移矩阵，用来描述传入坐标的轨迹
B = np.mat([[0.5], [1]])#加速度矩阵，暂时不用，设定为匀速运动
H = np.mat([[1, 0], [0, 1]])#预测转移矩阵，用来描述理想轨迹

# print('H=', H)
#噪声的初始化
Q = np.mat([[1, 0], [0, 1]])
R = np.mat([[10, 1], [1, 5]])
# print('R=', R)
# print('Q=', Q)
random2 = np.random.randn(2, N) #生成2xN的随机整数矩阵
W = np.dot(np.sqrt(Q), random2) #对矩阵进行求方根使用numpy库
V = np.dot(np.sqrt(R), random2)

X = np.zeros((2, N))
# print('s=', S.shape)
for i in range(0, N, 1):
    X[:, i] = S[i, :] #传入的坐标
P0 = np.mat([[1, 1], [1, 1]])#协方差矩阵
Z = np.zeros((2, N))#预测无人机的矩阵
Z[:, 0] = [X[0, 0], X[1, 0]] #观测初始化，将第一个传入坐标定为第一个预测坐标
Xkf = np.zeros((2, N))
Xkf[:, 0:1] = X[:, 0:1]#初始化的滤波坐标
I = np.identity(2)
for i in range(1, N, 1):
    Z[:, i:i+1] = H*X[:, i:i+1]+V[:, i:i+1]#预测的无人机轨迹坐标
for i in range(1, N, 1):
    X_pre = A*Xkf[:, i-1:i]
    P_pre = A*P0*A.T+Q

    Kg = P_pre*H.T*inv(H*P_pre*H.T+R)
    Xkf[:, i:i+1] = X_pre+Kg*(Z[:, i:i+1]-H*X_pre)
    P0 = (I-Kg*H)*P_pre
print('Xkf=', Xkf)

plt.figure()

# plt.subplot(131)
# plt.title('REAL')
# plt.plot(X[0, :], X[1, :], 'b+')
# plt.subplot(132)
# plt.title('FILTER')
# plt.plot(Xkf[0, :], Xkf[1, :], 'r+')
# plt.subplot(133)
# plt.title('IDEAL')
# plt.plot(Z[0, :], Z[1, :], 'y+')
plt.plot(X[0, :], X[1, :], 'b+')
plt.plot(Xkf[0, :], Xkf[1, :], 'r+')
plt.plot(Z[0, :], Z[1, :], 'g-')
print(Xkf)
plt.show()