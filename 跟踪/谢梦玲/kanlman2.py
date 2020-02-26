import numpy as np
import matplotlib.pyplot as plt
import math
import random
from numpy import random
from numpy import *
from numpy.linalg import*

Zs = zeros((3, 100))  # 目标到三个基站的距离
h = zeros((3, 100))  # 最终的预测方程算出来的坐标
Z = zeros((3, 100))  # 目标到主基站的距离

BS1 = np.mat([3000, 100, 20])
BS2 = np.mat([200, 3000, 50])
BS3 = np.mat([300, 400, 3000])

BSb = np.mat([0, 0, 0])
#计算距离
def Z_pre(X):
    # print('X[0, 1]:',X[0, 1])
    global Zs, h, Z
    global BS1, BS2, BS3, BSb

    # #改为列处理
    # Zs[:, 0] = sqrt((X[0, 0] - BS1[0, 0]) ** 2 + (X[0, 1] - BS1[0, 1]) ** 2 + (X[0, 2] - BS1[0, 2]) ** 2)
    # Zs[:, 1] = sqrt((X[0, 1] - BS2[0, 1]) ** 2 + (X[0, 1] - BS2[0, 1]) ** 2 + (X[0, 2] - BS2[0, 2]) ** 2)
    # Zs[:, 2] = sqrt((X[0, 2] - BS3[0, 2]) ** 2 + (X[0, 2] - BS3[0, 2]) ** 2 + (X[0, 2] - BS3[0, 2]) ** 2)
    # Z[:, 0] = sqrt((X[0, 0] - BSb[0, 0]) ** 2 + (X[0, 1] - BSb[0, 1]) ** 2 + (X[0, 2] - BSb[0, 2]) ** 2)
    # Z[:, 1] = sqrt((X[0, 0] - BSb[0, 1]) ** 2 + (X[0, 1] - BSb[0, 1]) ** 2 + (X[0, 2] - BSb[0, 2]) ** 2)
    # Z[:, 2] = sqrt((X[0, 0] - BSb[0, 0]) ** 2 + (X[0, 1] - BSb[0, 1]) ** 2 + (X[0, 2] - BSb[0, 2]) ** 2)
    # h[:, 0] = Zs[:, 0] - Z[:, 0]
    # h[:, 1] = Zs[:, 1] - Z[:, 1]
    # h[:, 2] = Zs[:, 2] - Z[:, 2]
    # return h

n = 5
t = 1
X = np.zeros((6, n))
X[:, 0:1] = np.mat([[100], [200], [100], [4], [8], [0.5]])
Z = np.zeros((3, n))
h = np.zeros((3, n))
A = np.mat([[1, 0, 0, t, 0, 0],
           [0, 1, 0, 0, t, 0],
           [0, 0, 1, 0, 0, t],
           [0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 1]])

dat = 1.35
r = np.array([1, 1, 1])
G = np.mat([[t**2/2, 0, 0],
           [0, t**2/2, 0],
           [0, 0, t**2/2],
           [t, 0, 0],
           [0, t, 0],
           [0, 0, t]])
Q = dat*np.diag(r)
TSOA = math.sqrt(10)
TDOA = math.sqrt(10)
random2 = random.randn(3, 1)
W = sqrt(Q)*random2

L = 6
alpha = 0.3#可以调节 改变均值
kalpha = 0.54
belta = 2#对高斯分布通常是2最优100 可以改变方差
lamada = alpha*alpha*(L+kalpha)-L
c = L+lamada
Wm = [lamada/c, 0.5/c, 0.5/c, 0.5/c, 0.5/c, 0.5/c, 0.5/c, 0.5/c, 0.5/c, 0.5/c, 0.5/c, 0.5/c, 0.5/c]
Wc = Wm
Wc[0] = Wc[0]+(1-alpha**2+belta)
c = sqrt(c)
xsP1 = zeros((6, 6))
xsP2 = zeros((6, 6))
xsP11 = zeros((6, 6))
xsP22 = zeros((6, 6))
ZS = zeros((3, n))
Xukf = zeros((6, n))
Xukf[:, 1] = X[:, 1]
P0 = eye(6)
c = sqrt(c)
for i in range(1, n):
    X[:, i:i+1] = A * X[:, i - 1:i] #状态方程，目标轨迹
print('X', X)
Z = Z_pre(Z)
print('Z', Z)
#     #算法
# for t in range(1, n):
#     xestimate = Xukf[:, t-1:t]
#     P = P0
#     cho = c * cholesky(P).T
#     #取SIGMA点 将原来的预测值作为样本均值
#     for k in range(0, L):
#          xsP1[:, k:k+1] = xestimate + cho[:, k:k+1]
#          xsP2[:, k:k+1] = xestimate - cho[:, k:k+1]
#     print('xsP1', xsP1)
#     print('xsP2', xsP1)
    # for k in range(0, L-1):
    #     sigma1 = np.hstack((xestimate, xsP1, xsP2))
    #     Xsigmapre = A * sigma1
    #     Xpred = zeros((6, 1))
    # # 对SIGMA点进行非线性变换
    # for K in range(0, 2*L+1):
    #     #计算输出量的均值与方差  Xpred为均值
    #   Xpred = Xpred + Xsigmapre[:, K]*Wm[K]
    # Ppred = zeros((6, 6))
    # for k in range(0, 2*L):
    #     #Ppred为协方差
    #      Ppred = Ppred + Wc[k] * (Xsigmapre[:, k] - Xpred)*(Xsigmapre[:, k] - Xpred).T
    # Ppred = Ppred + G * Q * G.T
    # chor = c * cholesky(Ppred).T
    # for k in range(0, 2*L-1):
    #      xsP11[:, k:k+1] = Xpred + chor[:, k:k+1]
    #      xsP22[:, k:k+1] = Xpred - chor[:, k:k+1]
    # #SIGMA2为非线性变换后的取值点
    # sigma2 = np.hstack([Xpred, xsP11, xsP22])
    # print('sigma2', sigma2)
    # H = zeros([4, 2*L+1])
    # for k in range(0, 2*L-2):
    #     mid = sigma2[:, k]
    #
    #     # print('mid:', mid)
    #     print('\n\n\n\n')

        # ZS[:, k] =Z_pre(mid)

# Zpred = 0
    # for k in range(0, 2*L):
    #     Zpred = Zpred + Wm[k] * ZS[:, k]
    # Pzz = zeros((3, 3))
    # for k in range(0, 2*L):
    #     Pzz = Pzz + W[k] * (ZS[:, k] - Zpred)*(ZS[:, k] - Zpred).T
    # Pzz = Pzz+r
    # Pxz = zeros((6, 3))
    # for k in range(0, 2*L):
    #     Pxz = Pxz + Wc[k] * (sigma1[:, k] - Xpred)*(ZS[:, k] - Zpred).T
    # K = Pxz * inv(Pzz)
    # xestimate = Xpred+K*(Z[:, t]-Zpred)
    # P = Ppred-K*Pzz*K.T
    # P0 = P
    # Xukf[:, t] = xestimate