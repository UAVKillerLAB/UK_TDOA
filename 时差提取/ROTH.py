# clear; clc;close all;
#注意，本代码目前的内容是仿真PHAT加权法而不是原来的ROTH，具体原因详见readme
import numpy as np
import matplotlib.pyplot as plt


def wgn(x, snr):
    P_signal = np.sum(abs(x) ** 2) / len(x)
    P_noise = P_signal / 10 ** (snr / 10.0)
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
