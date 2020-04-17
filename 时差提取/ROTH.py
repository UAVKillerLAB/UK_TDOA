# clear; clc;close all;
#注意，本代码目前的内容是仿真PHAT加权法而不是原来的ROTH，具体原因详见readme
import numpy as np
import matplotlib.pyplot as plt

'''
def wgn(x, snr):
    P_signal = np.sum(abs(x) ** 2) / len(x)
    P_noise = abs(P_signal / 10 ** (snr / 10.0))
    return np.random.randn(len(x)) * np.sqrt(P_noise)


N = 1024  # 采样点
FS = 5000000  # 采样频率
SNR = 20  # 信噪比设置

n = np.arange(0, N - 1)
d = 6
x1 = 5 * np.cos(2 * np.pi * 10 * n / FS)  # 输入函数1
# noise1 = wgn(x1, SNR)
# s1 = x1 + noise1
x2 = 5 * np.cos(2 * np.pi * 10 * (n + d) / FS)  # 输入函数2
# noise2 = wgn(x2, SNR)
# s2 = x2 + noise2

x1 = list(x1)
x2 = list(x2)
Ts = 1 / FS
Ta = np.arange(0, Ts * N, Ts / 10)


fa = list(range(0 , len(Ta)))
i = 0
for i in fa:
    fa[i] = 0

fb = list(range(0 , len(Ta)))
i = 0
for i in fb:
    fb[i] = 0

T = np.arange(0, len(Ta) - 1)
k = np.arange(0, Ts * N, Ts)
t = np.arange(0, len(k) - 1)

for Tx in T:
    for tx in t:
        fa[Tx] = fa[Tx] + x1[tx] * np.sinc((Tx / 10 - tx))

for Tx in T:
    for tx in t:
        fb[Tx] = fb[Tx] + x2[tx] * np.sinc((Tx / 10 - d - tx))

fa = np.array(fa)
fb = np.array(fb)

noise1 = wgn(fa, SNR)
noise2 = wgn(fb, SNR)
s1 = fa + noise1
s2 = fb + noise2

X1 = np.fft.fft(s1, 2 * N - 1)  # 快速傅里叶正变换处理
X2 = np.fft.fft(s2, 2 * N - 1)
Sxy = X1 * np.conj(X2)  # 取x1和x2的互相关函数

PA = np.fft.fftshift(np.fft.ifft(Sxy / (abs(Sxy))))
PA = PA.real

t1 = np.arange(-N + 1, N) / FS

# 以下为画图操作
plt.plot(t1, PA, 'r')
plt.title('PHAT')
plt.xlabel('t/s')
plt.ylabel('PA(t)')

RH_list = list(PA)
dis2 = RH_list.index(max(RH_list))  # 取峰值
k = max(RH_list)

d2 = dis2 - N
delay = d2 / FS  # 时延估计

print(delay)
plt.show()
'''

import numpy as np
import os

def wgn(x, snr):
    P_signal = np.sum(abs(x) ** 2) / len(x)
    P_noise = P_signal / 10 ** (snr / 10.0)
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

    x1 = need[2350:N + 2349] # 主要信号

    SNR = 20
    D = 6.04635194728013e-07
    D1 = 0 - 1.966503114547034e-07

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

    # noise1 = wgn(x1, SNR)
    # s1 = x1 + noise1
    # noise2 = wgn(x2, SNR)
    # s2 = x2 + noise2
    # noise3 = wgn(x3, SNR)
    # s3 = x3 + noise3
    #---------------
    s1 = x1
    s2 = x2
    s3 = x3
    # ---------------

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

    Delay12 = d12 / fs  # 时延估计
    Delay13 = d13 / fs

    print('主站和基站1的时差为')
    print(Delay12)
    print('主站和基站2的时差为')
    print(Delay13)
