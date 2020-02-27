import numpy as np
import matplotlib.pyplot as plt
import math
import random
from numpy import random
from numpy import *
from numpy.linalg import*
# from numpy import mat

T = 1
N = 5
x = np.zeros((6, N))

#目标的初始位置
x[:,0] = ([100, 200, 100, 4, 8, 0.5])
# print(X)
Z = np.zeros((3, N))
h = np.zeros((3, N))
#状态转换矩阵
A =np.matrix([[1,0,0,T,0,0],
              [0,1,0,0,T,0],
              [0,0,1,0,0,T],
              [0,0,0,1,0,0],
              [0,0,0,0,1,0],
              [0,0,0,0,0,1]])

BS1 = np.mat([3000, 100, 20])
BS2 = np.mat([200, 3000, 50])
BS3 = np.mat([300, 400, 3000])
BSb = np.mat([450, -200, 100])

dat = 1.35
Q = dat*np.eye(3, dtype = int)
# print('Q=',Q)
TSOA = math.sqrt(10)
TDOA = math.sqrt(10)
W = np.sqrt(Q)*random.randn(3, 1)
# print('W=',W)
R = np.diag([TDOA, TDOA, TDOA])
# print('R=',R)
G = np.array([[T*T/2.0, 0, 0],
              [0, T*T/2.0, 0],
              [0, 0, T*T/2.0]])
# G = np.eye(3,dtype = T^2/2)   错误代码语句
# print('G = ',G)

def  h_pre(x):
    global T, BS1, BS2, BS3, BSb
    # 距离公式
    # #输入三个数据，X（1）、X（2）、X（3）
    # print('输入X（1）：')
    # x1 = float(input())
    # print('输入X（2）：')
    # x2 = float(input())
    # print('输入X（3）：')
    # x3 = float(input())
    # x = np.zeros((1, 6))
    # x[:, 0] = x1
    # x[:, 1] = x2
    # x[:, 2] = x3
    # #print('x = ',x)
    h1 = np.zeros((1, 3))
    h1[:, 0] = math.sqrt((x[0] - BS1[0])**2 + (x[1] - BS1[1])**2 + (x[2] - BS1[2]**2))
    h1[:, 1] = math.sqrt((x[0] - BS2[0])**2 + (x[1] - BS2[1])**2 + (x[2] - BS2[2]**2))
    h1[:, 2] = math.sqrt((x[0] - BS3[0])**2 + (x[1] - BS3[1])**2 + (x[2] - BS3[2]**2))
    h2 = np.zeros((1, 3))
    h2[0:, ] = math.sqrt((x[0] - BSb[0])**2 + (x[1] - BSb[1])**2 + (x[2] - BSb[2]**2))
    h = h1 - h2
    f = np.array([x[0] + T*x[3]], [x[1] + T*x[4]], [x[2] + T*x[5]], [x[3]], [x[4]], [x[5]])

# 状态方程，目标轨迹
# while 2<=t<=N:
#     x[:, 1] = A*x[:, t-1]
#     t=t + 1

#观测方程，，目标观测
# while  1<=t<=N

L = 6
alpha = 0.3 #可调节
kalpha = 0.54
belta = 2 #高斯分布通常2为最优100
lamada = alpha**2*(L + kalpha) - L
c =  L + lamada
# Wm = mat(zeros((1, 2*L+1)))  :可用：from numpy import mat  实现
Wm = np.zeros((1, 2*L+1))
Wm[:, 0] = lamada/c
Wm[:, 1:2*L+1] = 1/(2*c)
# print('Wm = ',Wm)
Wc = Wm
# Wc[:, 0] =  Wc[0] + (1-alpha**2 + belta)  :Wc[0]的提取方式是错的  既然前面赋值的是第一行第一个，
#                                            那么提出来应该也是一行第一个，所以是Wc[:,0]
Wc[:, 0] =  Wc[:,0] + (1-alpha**2 + belta)
# print('Wc = ',Wc)
Xukf = np.zeros((6, N))
Xukf[:, 0] = x[:, 0]
P0 = np.eye(6, dtype = float)
c = math.sqrt(c)
