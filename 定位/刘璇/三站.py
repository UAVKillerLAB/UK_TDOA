from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from numpy.linalg import *

c = 3 * 10 ** 8
t1 = 0.00000176627409214165;
t2 = 0.00000240583312151407
t3 = 0.00000330487932593047
X = array([0, 750, -750, 0])
Y = array([0, 1229, 1229, -1500])
Z = array([1, 2, 1, 0])

fig = plt.figure(5)
ax = fig.add_subplot(1, 1, 1, projection='3d')

ax.scatter3D(X, Y, Z, c='b', marker='o')
ax.set_xlim(-1000, 1000)
ax.set_ylim(-2000, 2000)
ax.set_zlim(-1000, 1000)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.title(' Base Station')

deltaR = [c * t1, c * t2, c * t3]
K = [0] * 4
k = [0] * 3
for i in range(0, 4):
    K[i] = X[i] ** 2 + Y[i] ** 2 + Z[i] ** 2
for i in range(0, 3):
    k[i] = 0.5 * (deltaR[i] ** 2 + K[0] - K[i + 1])
A1 = [X[0] - X[1], Y[0] - Y[1], Z[0] - Z[1]]
A2 = [X[0] - X[2], Y[0] - Y[2], Z[0] - Z[2]]
A3 = [X[0] - X[3], Y[0] - Y[3], Z[0] - Z[3]]
A = array([A1, A2, A3])
aa = dot(inv(dot(A.T, A)), A.T)

m = [0] * 3
n = [0] * 3
for i in range(0, 3):
    m[i] = aa[i][0] * k[0] + aa[i][1] * k[1] + aa[i][2] * k[2]
    n[i] = aa[i][0] * deltaR[0] + aa[i][1] * deltaR[1] + aa[i][2] * deltaR[2]

a = n[0] ** 2 + n[1] ** 2 + n[2] ** 2 - 1
b = (m[0] - X[0]) * n[0] + (m[1] - Y[0]) * n[1] + (m[2] - Z[0]) * n[2]
c = (m[0] - X[0]) ** 2 + (m[1] - Y[0]) ** 2 + (m[2] - Z[0]) ** 2

delta = b ** 2 - a * c

t1 = (-b + sqrt(delta)) / a
R = array([0] * 4)
R[0] = t1
MS = array([0, 0, 0])
if (delta >= 0):
    MS[0] = m[0] + n[0] * R[0]
    MS[1] = m[1] + n[1] * R[0]
    MS[2] = m[2] + n[2] * R[0]
else:
    input("Wrong Position1")
ax.scatter3D(MS[0], MS[1], MS[2], c='r', marker='^')

for i in range(0, 4):
    R[i] = sqrt((MS[0] - X[i]) ** 2 + (MS[1] - Y[i]) ** 2 + (MS[2] - Z[i]) ** 2)
h = array([[deltaR[0] - (R[1] - R[0])], [deltaR[1] - (R[2] - R[0])], [deltaR[2] - (R[3] - R[0])]])
G1 = dot(ones((3, 3)) * (1 / R[0]), array([[X[0] - MS[0], Y[0] - MS[1], Z[0] - MS[2]], [0, 0, 0], [0, 0, 0]]))

for i in range(2, 5):
    P2 = (1 / R[1]) * array([X[1] - MS[0], Y[1] - MS[1], Z[1] - MS[2]])
    P3 = (1 / R[2]) * array([X[2] - MS[0], Y[2] - MS[1], Z[2] - MS[2]])
    P4 = (1 / R[3]) * array([X[3] - MS[0], Y[3] - MS[1], Z[3] - MS[2]])

G2 = array([P2, P3, P4])
Gt = G1 - G2
errors = array([[1], [2], [3]])
Q0 = dot(errors, errors.T)
Q = cov(Q0)
delta2 = dot(dot(dot(pinv(dot(dot(Gt.T, pinv(Q)), Gt)), Gt.T), pinv(Q)), h)
MSS = array([0] * 3)
if (abs(delta2[0]) + abs(delta2[1]) + abs(delta2[2]) > 0.5):
    for i in range(3):
        MSS[i] = MS[i] + delta2[i]
ax.scatter3D(MSS[0], MSS[1], MSS[2], c='c', marker='d')
plt.show()
