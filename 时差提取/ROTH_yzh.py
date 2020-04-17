
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal, fftpack
#----------------------------------------------------------------参数设定
N = 4096*2                          # 采样点
FS = 200e6                          # 采样频率

n = np.arange(0, N - 1)             #创建时间
t = n/FS
L = np.arange(-N, N+1)              #创建频率
d = 10                              #时差delta_t = d / FS
#----------------------------------------------------------------信号输入
signal_raw = np.fromfile("信号.bin",dtype = np.int16)             #读取信号文件的数据格式为16位
signal_1 = signal_raw / max(abs(signal_raw))                      #归一化
signal_2 = signal_raw / max(abs(signal_raw))

signal_a = signal_1[2350:N+2349]                                  #模拟时差
signal_b = signal_1[2350+d:N+2349+d]
#-----------------------------------------------------------------信号处理
Fx = np.fft.fft(signal_a , n = 2*N+1)                             #傅里叶变换
Fy = np.fft.fft(signal_b , n = 2*N+1)
Rxy = Fx*np.conj(Fy)                                              #两个信号共轭
Sxy = np.fft.fftshift(np.fft.ifft(Rxy))                           #反傅里叶变换

delay = np.argmax(np.array(Sxy)) - N                              # 返回Sxy内最大值位置，并输出时差
print(delay/FS)
#-----------------------------------------------------------------画图
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)                 #创建两张图标
ax1.plot(t, signal_a, 'r')
ax1.set_title("信号1时间谱")

ax2.plot(t, signal_b, 'b')
ax2.set_title("信号2时间谱")

plt.rcParams['font.sans-serif']=['SimHei']                        #显示中文标签
plt.rcParams['axes.unicode_minus'] = False                        #显示负号标签
plt.show()

plt.plot(L, abs(Sxy/max(abs(Sxy))), 'g')
plt.show()