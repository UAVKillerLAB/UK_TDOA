#此程序操作说明：运行，输入目标坐标，要修改基站坐标请在标记处修改（下文有标记），输出定位坐标和图像

import numpy as np
import math
import matplotlib.pyplot as plt

t = [0.0, 0.0]
# t = [4.58465298337831e-06, 0.01314005170586e-06]
t[0] = float(input('请输入时差1：'))
t[1] = float(input('请输入时差2：'))
# 变量输入
c = 3.0e8
# 光速
m = np.array([1.0, 1.0])  # 为最终的二次方程当中系数里面的一个量
n = np.array([1.0, 1.0])  # 为最终的二次方程当中系数里面的一个量
R = np.array([1.0, 1.0])  # 信号到其中一个副站和主基站之间的时间差
k = np.array([1.0, 1.0])  # 线性转换的中间变量
A = np.array([[1.0, 1.0], [1.0, 1.0]])  # 线性转换当中中间矩阵

#请在此处修改基站坐标
x = np.array([-866.0, 866.0, 0.0])  # 基站的x坐标
y = np.array([500.0, 500.0, -1000.0])  # 基站的y坐标

r = np.array([0.0, 0.0, 0.0])  # 为信号源到基站的距离差
B = np.array([[0.0, 0.0], [0.0, 0.0]])  # 线性转换中间变量
E = np.array([[0.0, 0.0], [0.0, 0.0]])  # 线性转换中间变化
h = np.array([0.0, 0.0])  # 中间变量
# 中间变量初始化
for i in range(0, 2, 1):
    R[i] = t[i] * c
# 定位距离差

for i in range(0, 2, 1):
    k[i] = 0.5 * (R[i] ** 2 + (x[0] ** 2 + y[0] ** 2) - (x[i + 1] ** 2 + y[i + 1] ** 2))

for i in range(0, 2, 1):
    A[i][0] = x[0] - x[i + 1]
    A[i][1] = y[0] - y[i + 1]

C = np.transpose(A)
D = np.linalg.inv(np.dot(C, A))
# 最小二乘法

F = np.dot(D, C)

for j in range(0, 2, 1):
    m[j] = F[j][0] * k[0] + F[j][1] * k[1]
    n[j] = F[j][0] * R[0] + F[j][1] * R[1]
# 中间线性转换步骤，取自无源定位相关论文
a = n[0] ** 2 + n[1] ** 2 - 1
b = (m[0] - x[0]) * n[0] + (m[1] - y[0]) * n[1]
c = (m[0] - x[0]) ** 2 + (m[1] - y[0]) ** 2
# 关于R0的二次方程系数
r0 = (-2 * b + math.sqrt(4 * b ** 2 - 4 * a * c)) / (2 * a)
# 求根公式
S = [m[0] + n[0] * r0, m[1] + n[1] * r0]
# 坐标
# print(S)
# for i in range(0, 3, 1):
#    plt.plot(x[i], y[i], 'ro')
# plt.plot(S[0], S[1], 'ro')
# plt.show()
n = [0.00000001, 0.00000002]
Q = np.cov(n)

for n in range(1, 10000):
    r[0] = ((S[0] - x[0]) ** 2 + (S[1] - y[0]) ** 2) ** 0.5
    r[1] = ((S[0] - x[1]) ** 2 + (S[1] - y[1]) ** 2) ** 0.5
    r[2] = ((S[0] - x[2]) ** 2 + (S[1] - y[2]) ** 2) ** 0.5
    B = [[x[0] - x[1], y[0] - y[1]], [x[0] - x[2], y[0] - y[2]]]
    E = B
    h = [t[0] - (r[1] - r[0]) / 3e8, t[1] - (r[2] - r[0]) / 3e8]
    h1 = np.transpose(h)
    V = np.transpose(E)
    N = 1 / Q
    U1 = np.dot(V, N, )
    U2 = np.dot(U1, E)
    U3 = np.linalg.inv(U2)
    U4 = np.dot(U3, V)
    U5 = np.dot(U4, N)  # U1到U5都只为实现最小二乘法的一个式子当中的中间变量，写法有些复杂
    k = np.dot(U5, h1)
    S = S + np.transpose(k)
    p = (k[0] ** 2 + k[1] ** 2) ** 0.5
    if p < 0.00000000000000000000001:
        break
# 泰勒循环迭代
print(S)
for i in range(0, 3, 1):
    plt.plot(x[i], y[i], 'ro')
plt.plot(S[0], S[1], 'bo')
plt.show()

# 上诉中间变量步骤都取自无人机无源定位相关论文当中运行过程
