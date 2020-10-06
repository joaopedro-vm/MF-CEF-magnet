#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 23:49:16 2020

@author: joaop

Calculate the entropy variation of DyAl2 with the adiabatic application of a
magnetic field, versus the temperature, for the field with different magnitudes
and directions.

It calculates the entropy with and without the field, and then subtracts
one from the other.
"""

# Import useful packages
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from MF_magnet_backend import *

##############################################################################

j = 15/2 # total angular momentum

prec = 1e-2
cut = 0

# Disprosium's saturation magnetization in easy magnetization direction (bohr magneton units)
easy = np.array([0,0,9.85])

# Values of magnetic fiedl magnitude (T)
h = np.array([0., 2, 5])

##############################################################################

# [001] field direction

# Number of temperature values
n = 50

# Temperature array
T = np.linspace(0, 140, n+1)[1:]

# Entropy array for different temperature and field values
S_T001 = np.zeros((n, len(h)))

# Array of magnetic field
H = np.zeros((len(h), 3))

# Defining magnetic field direction
H[:,2] = h

for i in range(n):
    for k in range(len(h)):
        
        m = sc_mag_min_F(H[k], j, T[i], prec, cut, easy, np.array([1.,1.,1.]))
        E = np.linalg.eigh(Hamiltonian(H[k], m, j))[0]
        S_T001[i, k] = S(E, T[i])
        
        print("H = %.2f, T <001>: %.2f%%" % (H[k,0], (i*len(h)+k+1)/(n*len(h))*100))

# Plotting DeltaS vs T results

plt.scatter(dyal2_0_2T[:,0], dyal2_0_2T[:,1], facecolors="none", edgecolors="red")
plt.plot(dyal2_0_2T_pred[:,0], dyal2_0_2T_pred[:,1], "--", color="red")
plt.scatter(dyal2_0_5T[:,0], dyal2_0_5T[:,1], facecolors="none", edgecolors="green")
plt.plot(dyal2_0_5T_pred[:,0], dyal2_0_5T_pred[:,1], "--", color="green")
plt.plot(T, -R/kB*(S_T001[:,1]-S_T001[:,0]), label=r"H: 0 $\rightarrow$ 2 T", color="red")
plt.plot(T, -R/kB*(S_T001[:,2]-S_T001[:,0]), label=r"H: 0 $\rightarrow$ 5 T", color="green")
plt.xlabel("T"); plt.ylabel("$-\Delta$S")
plt.legend()
plt.xlim(0, 140)
plt.hlines(0, 0, 140)
plt.savefig("dSvsT[001].png", format="png", dpi=300, bbox_to_inches=True)

##############################################################################

# [011] field direction

# Number of temperature values
n = 20

# Temperature array
T = np.linspace(0, 80, n+1)[1:]

# Entropy array for different temperature and field values
S_T011 = np.zeros((n, len(h)))

# Array of magnetic field
H = np.zeros((len(h), 3))

# Defining magnetic field direction
H[:,0] = H[:,1] = h

for i in range(n):
    for k in range(len(h)):
        
        m = sc_mag_min_F(H[k]/np.sqrt(2), j, T[i], prec, cut, easy, np.array([0.,1.,1.])) 
        E = np.linalg.eigh(Hamiltonian(H[k], m, j))[0]
        S_T011[i, k] = S(E, T[i])
        
        print("H = %.2f, T <001>: %.2f%%" % (H[k,0], (i*len(h)+k+1)/(n*len(h))*100))

# Plotting DeltaS vs T results

plt.plot(T, -(S_T011[:,1]-S_T011[:,0]), label="H: 0 $\rightarrow$ 2 T")
plt.plot(T, -(S_T011[:,2]-S_T011[:,0]), label="H: 0 $\rightarrow$ 5 T")
plt.xlabel("T")
plt.ylabel("$-\Delta$S")
plt.legend(); plt.xlim(0, 80)
plt.hlines(0, 0, 80)
plt.savefig("dSvsT[011].png", format="png", dpi=300, bbox_to_inches=True)

##############################################################################

# [111] field direction

# Number of temperature values
n = 40

# Temperature array
T = np.linspace(0, 80, n+1)[1:]

# Entropy array for different temperature and field values
S_T111 = np.zeros((n, len(h)))

# Array of magnetic field
H = np.zeros((len(h), 3))

# Defining magnetic field direction
H[:,0] = H[:,1] = H[:,2] = h

for i in range(n):
    for k in range(len(h)):
        
        m = sc_mag_min_F(H[k]/np.sqrt(3), j, T[i], prec, cut, easy, np.array([1.,1.,1.]))
        E = np.linalg.eigh(Hamiltonian(H[k]/np.sqrt(3), m, j))[0]
        S_T111[i, k] = S(E, T[i])
        
        print("H = %.2f T <001>: %.2f%%" % (H[k,0], (i*len(h)+k+1)/(n*len(h))*100))

##############################################################################

# Plotting DeltaS vs T results

plt.plot(T, -(S_T111[:,1]-S_T111[:,0]), label="H: 0 $\rightarrow$ 2 T")
plt.plot(T, -(S_T111[:,2]-S_T111[:,0]), label="H: 0 $\rightarrow$ 5 T")
plt.xlabel("T")
plt.ylabel("$-\Delta$S")
plt.legend(); plt.xlim(0, 80)
plt.hlines(0, 0, 80)
plt.savefig("dSvsT[111].png", format="png", dpi=300, bbox_to_inches=True)