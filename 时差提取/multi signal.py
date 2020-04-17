# clear; clc;close all;
import numpy as np


def wgn(x, snr):
    P_signal = np.sum(abs(x) ** 2) / len(x)
    P_noise = abs(P_signal / 10 ** (snr / 10.0))
    return np.random.randn(len(x)) * np.sqrt(P_noise)


N = 1024  # 采样点
FS = 5000000  # 采样频率
SNR = 20  # 信噪比设置

n = np.arange(0, N - 1)

# 参数设定
D1 = 0 # 主站延时点数
D2 = 6.04635194728013e-07 # 基站1延时点数
D3 = -1.966503114547034e-07 # 基站2延时点数

A1 = 5 # 主站信号幅值
A2 = 5 # 基站1信号幅值
A3 = 5 # 基站2信号幅值

f1 = 10 # 主站信号周期
f2 = 10 # 基站1信号周期
f3 = 10 # 基站2信号幅值

x1 = A1 * np.cos(2 * np.pi * f1 * (n + D1) / FS)  # 主站信号
noise1 = wgn(x1, SNR)
s1 = x1 + noise1

x2 = A2 * np.cos(2 * np.pi * f2 * (n + D2) / FS)  # 基站1信号
noise2 = wgn(x2, SNR)
s2 = x2 + noise2

x3 = A3 * np.cos(2 * np.pi * f3 * (n + D3) / FS)  # 基站2信号
noise3 = wgn(x3, SNR)
s3 = x3 + noise3


X1 = np.fft.fft(s1, 2 * N - 1)  # 快速傅里叶正变换处理
X2 = np.fft.fft(s2, 2 * N - 1)
X3 = np.fft.fft(s3, 2 * N - 1)

S12 = X1 * np.conj(X2)  # 取X1和X2的互相关函数
S13 = X1 * np.conj(X3)


PA12 = np.fft.fftshift(np.fft.ifft(S12 / (abs(S12 + np.spacing(1)))))
PA12 = PA12.real

PA13 = np.fft.fftshift(np.fft.ifft(S13 / (abs(S13 + np.spacing(1)))))
PA13 = PA13.real


RH_12 = list(PA12)
RH_13 = list(PA13)

dis12 = RH_12.index(max(RH_12))  # 取峰值

k = max(RH_12) # 调试专用

dis13 = RH_13.index(max(RH_13))  # 取峰值

d12 = dis12 - N
Delay12 = d12 / FS  # 时延估计

d13 = dis13 - N
Delay13 = d13 / FS  # 时延估计

print('主站和基站1的时差为')
print(Delay12)
print('主站和基站2的时差为')
print(Delay13)
