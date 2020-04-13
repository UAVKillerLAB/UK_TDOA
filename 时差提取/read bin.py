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

    x1 = need[2350:N + 2349] # 主站信号

    SNR = 20
    # 输入接口数值
    D  = 5.323635820946877e-07  # 基站1延时点数
    D1 = 1.0282142542749676e-06 # 基站2延时点数

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
