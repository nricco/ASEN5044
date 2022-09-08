# -*- coding: utf-8 -*-
"""
ASEN-5044 HW#1, Problem 3 

State-space matrix for an orbiting satellite
 
"""
import numpy as np
import matplotlib.pyplot as plt 
from scipy.linalg import expm 

debug = True
# Params and IC
Ix, Iy, Iz = 500, 750, 1000 # kgm**3 
p0 = 20 # rad/s
x0 = np.array([0, 0.1, 0]).T

dt = 0.1 # s
t0,t1 = 0.0, 5.0
step = int(t1/dt)+1
time = np.linspace(t0,t1,step)
if debug: print('time:\n',time)

# State space form matricies 
A = np.array([[0, 0, 0],[0, 0, p0*(Ix-Iz)/Iy],[0, p0*(Iy-Ix)/Iz, 0]])
B = np.array([[1/Ix, 0, 0],[0, 1/Iy, 0],[0, 0, 1/Iz]])
C = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])
D = np.zeros_like([B])

#
STM = expm(A*dt)
print("State Transitional Matrix:")
print(STM)

# Initialize empty lists to receive STM and state information
x_t = []
delp = []
delq = []
delr = []

for i,t in enumerate(time):
    x_t.append(list(np.matmul(expm(A*t),x0)))
    delp.append(x_t[i][0])
    delq.append(x_t[i][1])
    delr.append(x_t[i][2])

if debug: print('x(t)=e^At |0-5sec:\n',x_t)
plt.plot(time, delp,label='delp')
plt.plot(time, delq,label='delq')
plt.plot(time, delr,label='delr')
plt.legend()
plt.xlabel('Time (sec)')
plt.ylabel('x(t)')
plt.title('Satellite Perturbation Response')
plt.show()
    

