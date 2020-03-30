from matplotlib import pyplot as plot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
import matplotlib.pyplot as plt

A = np.array([[0, 750, -750, 0], [0, 1229, 1229, -1500], [1, 2, 1, 0]])  # 四个站

B = [[A[0][0] - A[0][1], A[1][0] - A[1][1], A[2][0] - A[2][1]],
     [A[0][0] - A[0][2], A[1][0] - A[1][2], A[2][0] - A[2][2]],
     [A[0][0] - A[0][3], A[1][0] - A[1][3], A[2][0] - A[2][3]]]  # 对原始定位方程进行转换时的中间矩阵

C = np.transpose(B)  # B的转置矩阵
D = np.linalg.inv(np.dot(C, B))
# F = np.dot(D,np.linalg.inv(B))  # F为系数矩阵的违逆矩阵
F = np.dot(D, C)
# t1 = float(input('Input the first jet lag:'))
# t2 = float(input('Input the second jet lag:'))
# t3 = float(input('Input the third jet lag:'))
# t = [t1, t2, t3]
# t = [4.69028027700184e-6, 4.36366217893204e-6,]
t = [0.0, 0.0, 0.0]
t[0] = float(input('请输入时差1：'))
t[1] = float(input('请输入时差2：'))
t[2] = float(input('请输入时差3：'))
# 参数输入
c = 3e8
# 光速
deltaR1 = c * t[0]
deltaR2 = c * t[1]
deltaR3 = c * t[2]
# 定位距离之差
K1 = 0.5 * (deltaR1 ** 2 + A[0][0] ** 2 + A[1][0] ** 2 + A[2][0] ** 2 - A[0][1] ** 2 - A[1][1] ** 2 - A[2][1] ** 2)
K2 = 0.5 * (deltaR2 ** 2 + A[0][0] ** 2 + A[1][0] ** 2 + A[2][0] ** 2 - A[0][2] ** 2 - A[1][2] ** 2 - A[2][2] ** 2)
K3 = 0.5 * (deltaR3 ** 2 + A[0][0] ** 2 + A[1][0] ** 2 + A[2][0] ** 2 - A[0][3] ** 2 - A[1][3] ** 2 - A[2][3] ** 2)
# 中间线性转换步骤
m1 = F[0][0] * K1 + F[0][1] * K2 + F[0][2] * K3
m2 = F[1][0] * K1 + F[1][1] * K2 + F[1][2] * K3
m3 = F[2][0] * K1 + F[2][1] * K2 + F[2][2] * K3
# 中间线性转换步骤
n1 = F[0][0] * deltaR1 + F[0][1] * deltaR2 + F[0][2] * deltaR3
n2 = F[1][0] * deltaR1 + F[1][1] * deltaR2 + F[1][2] * deltaR3
n3 = F[2][0] * deltaR1 + F[2][1] * deltaR2 + F[2][2] * deltaR3
# 中间线性转换步骤
f = n1 ** 2 + n2 ** 2 + n3 ** 2 - 1  # R0^2的系数
g = (m1 - A[0][0]) * n1 + (m2 - A[1][0]) * n2 + (m3 - A[2][0]) * n3  # R0系数参数部分
h = (m1 - A[0][0]) ** 2 + (m2 - A[1][0]) ** 2 + (m3 - A[2][0]) ** 2  # 常数部分

r0 = (-2 * g + math.sqrt(4 * g ** 2 - 4 * f * h)) / (2 * f)
# 求根公式
S = [m1 + n1 * r0, m2 + n2 * r0, m3 + n3 * r0]
# 坐标
print(S)
# 输出坐标
figure = plot.figure()
axes = Axes3D(figure)
X = np.arange(-300, 300, 1)
Y = np.arange(-300, 300, 1)
X, Y = np.meshgrid(X, Y)
for i in range(0, 4, 1):
    axes.scatter(A[0][i], A[1][i], A[2][i], c='b')  # 目标点蓝色
axes.scatter(S[0], S[1], S[2], c='r')  # 基站位置红色
plt.show()
# 描点

# 上诉中间变量步骤都取自无人机无源定位相关论文当中运行过程
