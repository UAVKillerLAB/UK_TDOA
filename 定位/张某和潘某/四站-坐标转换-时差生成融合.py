from matplotlib import pyplot as plot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
import matplotlib.pyplot as plt

#地球坐标系转空间坐标系部分
a = 6378137  # 长半轴
b = 6356752.3142  # 短半轴

E = math.sqrt(a * a - b * b) / a


def get_coordinate(latitude, longitude, altitude):
    B = math.radians(latitude);
    L = math.radians(longitude);
    H = altitude;
    f = 1 / 298.257223563;
    r = 6378137;
    b = r * (1 - f);

    e = math.sqrt(2 * f - f * f);

    N = r / math.sqrt(1 - e * e * math.sin(B) * math.sin(B));
    data = [(N + H) * math.cos(B) * math.cos(L), (N + H) * math.cos(B) * math.sin(L),
            (N * (1 - e * e) + H) * math.sin(B)];
    return data;


#定位部分
#x =0 #float(input('请输入坐标x：'))
#y =0 #float(input('请输入坐标y：'))
#地球坐标请于此处修改，P0为目标，P1为主站
P1 = get_coordinate(31.2494976200,121.4557457000,562)
P2 = get_coordinate(31.2445077800,121.4605522200,562)
P3 = get_coordinate(31.2441775600,121.4507675200,562)
P4 = get_coordinate(31.2536067000,121.4557457000,561)
P0 = get_coordinate(31.2473696300,121.4557457000,562)

print(P1)
c = 3.0e8
a=0.0

r1 =( (P1[0]-P0[0])**2 + (P1[1]-P0[1])**2 + (P1[2]-P0[2])**2)**0.5  #无人机到基站1的距离
r2 =( (P2[0]-P0[0])**2 + (P2[1]-P0[1])**2 + (P2[2]-P0[2])**2)**0.5   #无人机到基站2的距离
r3 =( (P3[0]-P0[0])**2 + (P3[1]-P0[1])**2 + (P3[2]-P0[2])**2)**0.5    #无人机到基站3的距离
r4 =( (P4[0]-P0[0])**2 + (P4[1]-P0[1])**2 + (P4[2]-P0[2])**2)**0.5      #无人机到基站4的距离
#if r1<r2:
#    a = r1
#    r1 = r2
#    r2 = a

#if r3<r2:
#    a = r3
#    r3 = r2
#    r2 = a

t1 = (r1/c) - (r2/c)
t2 = (r1/c) - (r3/c)
t3 = (r1/c) - (r4/c)



#A = np.array([[0, 750, -750, 0], [0, 1229, 1229, -1500], [1, 2, 1, 0]])  # 四个站
A = np.array([[P1[0], P2[0], P3[0], P4[0]], [P1[1], P2[1], P3[1], P4[1]], [P1[2], P2[2], P3[2], P4[2]]])
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
t[0] = t1#float(input('请输入时差1：'))
t[1] = t2#float(input('请输入时差2：'))
t[2] = t3#float(input('请输入时差3：'))
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
print('空间直角坐标系为：')
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

#空间坐标系转地球坐标系部分
a = 6378137
b = 6356752.3142

E2 = (a * a - b * b) / (b * a)

def transform_xyz2lonlathei(x, y, z):
    lon = math.degrees(math.atan2(y, x))

    S = math.atan2(z * a, math.sqrt(x * x + y * y) * b)

    lat = math.atan2(z + E2 * b * math.pow(math.sin(S), 3),(math.sqrt(x * x + y * y) - E * E * a * math.pow(math.cos(S), 3)))

    W = math.sqrt(1 - E * E * math.sin(lat) * math.sin(lat))
    N = a / W
    hei = math.sqrt(x * x + y * y) / math.cos(lat) - N

    lat = math.degrees(lat)
    #lat = lat + 0.000137417748183
    #lon = lon - 0.00002146012843
    return lat, lon, hei

print('地球坐标系坐标为（经度，纬度，海拔）：')
print(transform_xyz2lonlathei(S[0],S[1],S[2]))
print('三个时差为：')
print(t1)
print(t2)
print(t3)