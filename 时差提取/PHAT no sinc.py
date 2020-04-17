# clear; clc;close all;
import numpy as np
import matplotlib.pyplot as plt


def wgn(x, snr):
    P_signal = np.sum(abs(x) ** 2) / len(x)
    P_noise = abs(P_signal / 10 ** (snr / 10.0))
    return np.random.randn(len(x)) * np.sqrt(P_noise)


N = 1024  # 采样点
FS = 5000000  # 采样频率
SNR = 20  # 信噪比设置

n = np.arange(0, N - 1)
d = 6.04635194728013e-07
x1 = 5 * np.cos(2 * np.pi * 10 * n / FS)  # 输入函数1
noise1 = wgn(x1, SNR)
s1 = x1 + noise1
x2 = 5 * np.cos(2 * np.pi * 10 * (n + d) / FS)  # 输入函数2
noise2 = wgn(x2, SNR)
s2 = x2 + noise2


X1 = np.fft.fft(s1, 2 * N - 1)  # 快速傅里叶正变换处理
X2 = np.fft.fft(s2, 2 * N - 1)
Sxy = X1 * np.conj(X2)  # 取x1和x2的互相关函数

PA = np.fft.fftshift(np.fft.ifft(Sxy / (abs(Sxy + np.spacing(1)))))
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