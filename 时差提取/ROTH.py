# clear; clc;close all;
import numpy as np
import matplotlib.pyplot as plt
import matlab


def wgn(x, snr): #加噪函数
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)


N = 2048 #采样点
FS = 500 #采样频率
SNR = 10 #信噪比设置

n = np.arange(0, N - 1)
x1 = 5 * np.cos(2 * np.pi * 10 * n / FS) #输入函数1
x1 = wgn(x1, SNR)
x2 = 5 * np.cos(2 * np.pi * 10 * (n + 6) / FS) #输入函数2
x2 = wgn(x2, SNR)

X1 = np.fft.fft(x1, 2 * N - 1) #快速傅里叶正变换处理
X2 = np.fft.fft(x2, 2 * N - 1)
Sxy = X1 * np.conj(X2) #取x1和x2的互相关函数
R11 = X1 * np.conj(X1) #取x1的自相关函数
RH = np.fft.ifftshift(np.fft.ifft(Sxy / R11)) #ROTH加权处理
t1 = np.arange(-N, N - 1) / FS

#以下为画图操作
plt.plot(t1, RH, )
plt.title('ROTH加权')
plt.xlabel('t/s')
plt.ylabel('RH(t)')

max2 = RH.argmax()
RH_list = list(RH)
dis2 = RH_list.index(max(RH_list)) #取峰值
d2 = dis2 - N
delay = d2 / FS #时延估计
print(delay)
plt.show()
print(delay)