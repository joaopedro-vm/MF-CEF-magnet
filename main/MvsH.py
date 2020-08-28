#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 23:27:02 2020

@author: joaop

Calculate the magnetization of DyAl2 for different magnitudes and directions
of the applied magnetic field.
"""

# Import useful packages
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from MF_magnet_backend import *

##############################################################################

j = 15/2 # total angular momentum
T = 4.2 # temperature

# Disprosium's saturation magnetization in easy magnetization direction (bohr magneton units)
easy = np.array([0,0,9.85])

prec = 1e-2
cut = 0

# Number of magnetic field values
n = 50

# Magnetic field magnitude array
h = np.linspace(0, 10, n+1)[1:]

##############################################################################

# [001] field direction

M001 = np.zeros((n, 3)) # Array of magnetization vector
MH001 = np.zeros(n) # Array of magnitude of magnetization in fields' direction

# Array of magnetic field
H = np.zeros((n, 3))

# Defining magnetic field direction
H[:,2] = h

# Magnetization calculation
for i in range(len(h)):
    
    M001[i] = sc_mag_min_F(H[i], j, T, prec, cut, easy, np.array([0,0,1.]))
    
    E = np.linalg.eigh(Hamiltonian(H[i], M001[i], j))[0]
    MH001[i] = mh(np.array([0.,0.,1.]), M001[i])
    
    print("<001>: %.2f%%" % ((i+1)/len(h)*100))

##############################################################################

# [011] field direction
M011 = np.zeros((n, 3))
MH011 = np.zeros(n)

H = np.zeros((n, 3))

# Defining magnetic field direction
H[:,1] = H[:,2] = h

# Magnetization calculation
for i in range(len(h)):
    
    M011[i] = sc_mag_min_F(H[i]/np.sqrt(2), j, T, prec, cut, easy, np.array([0,1,1.]))
    
    E = np.linalg.eigh(Hamiltonian(H[i], M011[i], j))[0]
    MH011[i] = mh(np.array([0.,1.,1.]), M011[i])
    
    print("<011>: %.2f%%" % ((i+1)/len(h)*100))

##############################################################################

# [111] field direction
M111 = np.zeros((n, 3))
MH111 = np.zeros(n)

H = np.zeros((n, 3))

# Defining magnetic field direction
H[:,0] = H[:,1] = H[:,2] = h

# Magnetization calculation
for i in range(len(h)):
    
    M111[i] = sc_mag_min_F(H[i]/np.sqrt(3), j, T, prec, cut, easy, np.array([1,1,1.]))
    
    E = np.linalg.eigh(Hamiltonian(H[i]/np.sqrt(3), M111[i], j))[0]
    MH111[i] = mh(np.array([1.,1.,1.]), M111[i])
    
    print("<111>: %.2f%%" % ((i+1)/len(h)*100))

##############################################################################

# Plotting M vs H results

plt.figure(figsize=(10, 7))
plt.rcParams.update({"font.size":16})
plt.plot(h, MH001/muB, label=r"$\langle 001 \rangle$")
plt.plot(h, MH011/muB, label=r"$\langle 011 \rangle$")
plt.plot(h, MH111/muB, label=r"$\langle 111 \rangle$")
plt.scatter(k100[:,0], k100[:,1])
plt.scatter(k110[:,0], k110[:,1])
plt.scatter(k111[:,0], k111[:,1])
plt.legend()
plt.xlim(0, 10); plt.ylim(5, 11)
plt.xlabel("B (T)", fontsize=20)
plt.ylabel(r"M ($\mu_{B}$)", fontsize=20)
plt.show()

##############################################################################

# Total angular momentum angle, tan(theta) = sqrt(M_x^2+M_y^2) / M_z
def theta(m):
    
    t = np.zeros(len(m))
    
    for i in range(len(t)):
        
        a = m[i][np.where(m[i] == np.max(m[i]))[0][0]]
        a_i = np.where(m[i] == np.max(m[i]))[0][0]
        b, c = m[i][np.where([0,1,2] != a_i)]
        t[i] = np.arctan2(np.sqrt(b**2+c**2), a)
        
    return t*180/np.pi

plt.figure(figsize=(10, 7))
plt.rcParams.update({"font.size":16})
plt.plot(h, theta(M111))
plt.scatter(*theta_B.T)
plt.xlim(0, 10); plt.ylim(0, 60)
plt.xlabel("B (T)", fontsize=20)
plt.ylabel(r"$\Theta(B)$", fontsize=20)
plt.show()