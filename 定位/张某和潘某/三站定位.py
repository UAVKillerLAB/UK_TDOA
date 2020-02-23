import numpy as np
import math
import matplotlib.pyplot as plt

t=[0.0,0.0]
#t = [4.58465298337831e-06, 0.01314005170586e-06]
t[0] = float(input('请输入时差1：'))
t[1] = float(input('请输入时差2：'))
c = 3.0e8
m = np.array([1.0, 1.0])
n = np.array([1.0, 1.0])
R = np.array([1.0, 1.0])
k = np.array([1.0, 1.0])
A = np.array([[1.0, 1.0], [1.0, 1.0]])
x = np.array([-866.0, 866.0, 0.0])
y = np.array([500.0, 500.0, -1000.0])
r = np.array([0.0, 0.0, 0.0])
B = np.array([[0.0, 0.0], [0.0, 0.0]])
E = np.array([[0.0, 0.0], [0.0, 0.0]])
h = np.array([0.0, 0.0])
for i in range(0, 2, 1):
    R[i] = t[i]*c


for i in range(0, 2, 1):
    k[i] = 0.5*(R[i]**2+(x[0]**2+y[0]**2)-(x[i+1]**2+y[i+1]**2))

for i in range(0, 2, 1):
    A[i][0] = x[0] - x[i + 1]
    A[i][1] = y[0] - y[i + 1]

C = np.transpose(A)
D = np.linalg.inv(np.dot(C, A))


F = np.dot(D, C)

for j in range(0, 2, 1):
    m[j] = F[j][0]*k[0]+F[j][1]*k[1]
    n[j] = F[j][0]*R[0]+F[j][1]*R[1]

a = n[0]**2+n[1]**2-1
b = (m[0]-x[0])*n[0]+(m[1]-y[0])*n[1]
c = (m[0]-x[0])**2+(m[1]-y[0])**2

r0 = (-2*b + math.sqrt(4*b**2-4*a*c))/(2*a)

S = [m[0]+n[0]*r0, m[1]+n[1]*r0]

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
    B = [[x[0]-x[1], y[0]-y[1]], [x[0]-x[2], y[0]-y[2]]]
    E = B
    h = [t[0]-(r[1]-r[0])/3e8, t[1]-(r[2]-r[0])/3e8]
    h1 = np.transpose(h)
    V = np.transpose(E)
    N = 1/Q
    U1 = np.dot(V, N,)
    U2 = np.dot(U1, E)
    U3 = np.linalg.inv(U2)
    U4 = np.dot(U3, V)
    U5 = np.dot(U4, N)
    k = np.dot(U5, h1)
    S = S+np.transpose(k)
    p = (k[0]**2+k[1]**2)**0.5
    if p < 0.00000000000000000000001:
        break

print(S)
for i in range(0, 3, 1):
   plt.plot(x[i], y[i], 'ro')
plt.plot(S[0], S[1], 'bo')
plt.show()