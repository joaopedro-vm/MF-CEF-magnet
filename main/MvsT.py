#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 18:38:50 2020

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

# Number of temperature values
n = 200

# Temperature array
T = np.linspace(0, 100, n+1)[1:]

##############################################################################

# [001] field direction

M = np.zeros((n, 3)) # Array of magnetization vector
Mmag = np.zeros(n) # Array of magnitude of magnetization in fields' direction

# Magnetization calculation
for i in range(len(T)):
    
    M[i] = sc_mag_min_F(np.zeros(3), j, T[i], prec, cut, easy, np.array([0,0,1.]))
    
    # E = np.linalg.eigh(Hamiltonian(np.zeros(0), M[i], j))[0]
    
    Mmag[i] = np.linalg.norm(M[i])
    
    print("%.2f%%" % ((i+1)/len(T)*100))

##############################################################################

Tc = T[np.where(np.diff(Mmag) == np.min(np.diff(Mmag)))[0][0]]

# Plotting M vs H results

plt.figure(figsize=(10, 7))
plt.rcParams.update({"font.size":16})
plt.plot(T, Mmag/muB)
plt.vlines(Tc, 0, np.max(Mmag)/muB*1.1)
plt.annotate("%.2f K" % (Tc), ((Tc+1)/100, 0.9), xycoords="axes fraction")
plt.xlim(0, 100)
plt.ylim(0, np.max(Mmag)/muB*1.1)
plt.xlabel("T(K)", fontsize=20)
plt.ylabel(r"M ($\mu_{B}$)", fontsize=20)
plt.savefig("MvsT.png", format="png", dpi=300, bbox_to_inches=True)