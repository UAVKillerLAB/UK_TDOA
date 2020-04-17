import numpy as np
import os
import math
import matplotlib.pyplot as plt
#                                                    以下为时差生成部分
x = float(input('请输入坐标x：'))
y = float(input('请输入坐标y：'))
#z = float(input('请输入坐标z：'))

c = 3.0e8
a = 0.0

#此处为4站布站
#P1 = [0,0,1]
#P2 = [750,1229,2]
#P3 = [-750,1229,1]
#P4 = [0,-1500,0]

#此处为4站的坐标运算
#r1 =( (P1[0]-x)**2 + (P1[1]-y)**2 + (P1[2]-z)**2)**0.5  #无人机到基站1的距离
#r2 =( (P2[0]-x)**2 + (P2[1]-y)**2 + (P2[2]-z)**2)**0.5   #无人机到基站2的距离
#r3 =( (P3[0]-x)**2 + (P3[1]-y)**2 + (P3[2]-z)**2)**0.5    #无人机到基站3的距离
#r4 =( (P4[0]-x)**2 + (P4[1]-y)**2 + (P4[2]-z)**2)**0.5      #无人机到基站4的距离

#此处为3站的布站方式
P1 = [-866,500]#此处是布站方式，要修改布站方式请在此处修改
P2 = [866,500]#
P3 = [0,-1000]#

#此处为3站布站运算
r1 =( (P1[0]-x)**2 + (P1[1]-y)**2)**0.5  #无人机到基站1的距离
r2 =( (P2[0]-x)**2 + (P2[1]-y)**2)**0.5   #无人机到基站2的距离
r3 =( (P3[0]-x)**2 + (P3[1]-y)**2)**0.5    #无人机到基站3的距离
#r4 =( (P4[0]-x)**2 + (P4[1]-y)**2)**0.5      #无人机到基站4的距离
#if r1<r2:
#    a = r1
#    r1 = r2
#    r2 = a

#if r3<r2:
#    a = r3
#    r3 = r2
#    r2 = a

#4站就去掉第3个注释
t1 = (r1/c) - (r2/c)
t2 = (r1/c) - (r3/c)
#t3 = (r1/c) - (r4/c)

#四站就去掉第3个注释
print('理论时差如下')
print(t1)
print(t2)
#print(t3)

#                                                   以下为时差提取部分

def wgn(x, snr):
    P_signal = np.sum(abs(x) ** 2) / len(x)
    P_noise = abs(P_signal / 10 ** (snr / 10.0))
    return np.random.randn(len(x)) * np.sqrt(P_noise)

if __name__ == '__main__':
    need = []
    filepath ='D:\信号.bin'
    binfile = open(filepath, 'rb') # 打开二进制文件
    size = os.path.getsize(filepath) # 获得文件大小
    print('size:', size/2)
    for i in range(int(size/2)):
        data = int.from_bytes(binfile.read(2), byteorder='little', signed=True) # 每次输出一个字节
        need.append(data)
    binfile.close()
    N = 2048
    fs = 200e6
    t = []
    for i in range(N-1):
        t.append(i/fs)
    y1 = np.array(need) / max(map(abs, need)) # 归一化

    x1 = need[2350:N + 2349] # 主站信号

    SNR = 20
    # 输入接口数值
    D  = t1  # 基站1延时点数
    D1 = t2 # 基站2延时点数

    N1 = D  / (1 / fs)
    N2 = D1 / (1 / fs)
    N1 = int(N1)
    N2 = int(N2)
    x2 = 0
    x3 = 0

    if N1 > 0:
        x2 = need[2350 - N1:N + 2349 - N1]
    if N1 < 0:
        x2 = need[2350 + N1:N + 2349 + N1]

    if N2 > 0:
        x3 = need[2350 - N2:N + 2349 - N2]
    if N2 < 0:
        x3 = need[2350 + N2:N + 2349 + N2]

    x1 = np.array(x1)
    x2 = np.array(x2)
    x3 = np.array(x3)

    noise1 = wgn(x1, SNR)
    s1 = x1 + noise1
    noise2 = wgn(x2, SNR)
    s2 = x2 + noise2
    noise3 = wgn(x3, SNR)
    s3 = x3 + noise3

    X1 = np.fft.fft(s1, 2 * N - 1)  # 快速傅里叶正变换处理
    X2 = np.fft.fft(s2, 2 * N - 1)
    X3 = np.fft.fft(s3, 2 * N - 1)
    S12 = X1 * np.conj(X2)  # 取x1和x2的互相关函数
    S13 = X1 * np.conj(X3)

    PA12 = np.fft.fftshift(np.fft.ifft(S12 / (abs(S12 + np.spacing(1)))))
    PA12 = PA12.real
    R12_list = list(PA12)

    PA13 = np.fft.fftshift(np.fft.ifft(S13 / (abs(S13 + np.spacing(1)))))
    PA13 = PA13.real
    R13_list = list(PA13)

    dis12 = R12_list.index(max(R12_list))  # 取峰值
    d12 = dis12 - N

    dis13 = R13_list.index(max(R13_list))  # 取峰值
    d13 = dis13 - N
    # 输出接口值
    Delay12 = d12 / fs  # 主站和基站1的时差值
    Delay13 = d13 / fs  # 主站和基站2的时差值

    if Delay12 * D < 0:
        Delay12 = -Delay12
    else:
        Delay12 = Delay12
    if Delay13 * D1 < 0:
        Delay13 = -Delay13
    else:
        Delay13 = Delay13

    print('主站和基站1的时差为')
    print(Delay12)
    print('主站和基站2的时差为')
    print(Delay13)
#                                                       以下为定位部分

t = [0.0, 0.0]
# t = [4.58465298337831e-06, 0.01314005170586e-06]
t[0] = Delay12
t[1] = Delay13
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