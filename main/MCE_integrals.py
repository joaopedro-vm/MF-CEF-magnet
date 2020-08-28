#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 00:35:34 2020

@author: joaop

I. Calculate the entropy variation of DyAl2 with the adiabatic application of a
magnetic field, versus the temperature, for the field with different magnitudes
in the [001] direction.
II. Calculate the heat capacity of DyAl2 for different temperatures

It calculates the magnetization for many values of temperature and magnetic
field, and then integrate the thermodynamic relations.
"""

# Import useful packages
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from MF_magnet_backend import *

##############################################################################

# Disprosium's saturation magnetization in easy magnetization direction (bohr magneton units)
easy = np.array([0,0,9.85])

# Temperature array
T = np.linspace(0, 140, 200)[1:]
dT = T[1] - T[0] # Temperature differential

# Magnetic field magnitude array
h = np.linspace(0, 5, 26)

# Magnetic field vector array
H = np.zeros((len(h), 3))

# Defining magnetic field direction
H[:,2] = np.copy(h)
dH = H[1] - H[0] # Magnetic field vector differential

# Magnetization array for different temperatures and fields
m = np.zeros((len(T), len(h), 3))

# Magnetization derivative by temperature array for different temperatures and fields
dmdT = np.zeros((len(T), len(h), 3))

# Energy array 
E = np.zeros((len(T), len(h)))

##############################################################################

# Magnetization calculation for different temperatures and fields
for i in range(len(T)):
    for k in range(len(H)):
        
        m[i,k] = sc_mag_min_F(H[k], j, T[i], prec, cut, np.array([0,0,9.85]), np.array([1.,1,1]))
        print(i*len(H)+k+1, "/", len(T)*len(h))

# Magnetization derivative by temperature
for i in range(len(H)):
     
    dmdT[:,i,:] = np.gradient(m[:,i,:], T, axis=0)
    
    # Energy calculation for future heat capacity calculation
    for k in range(len(T)):
        
        e = np.linalg.eigh(Hamiltonian(H[i], m[k,i], j))[0]
        E[k,i] = U(e, T[k])
    
dS_5T = np.sum(np.dot(dmdT, dH), axis=1)
dS_2T = np.sum(np.dot(dmdT[:,:11,:], dH), axis=1)

# Plotting DeltaS vs T results

plt.scatter(dyal2_0_2T[:,0], dyal2_0_2T[:,1], facecolors="none", edgecolors="red")
plt.plot(dyal2_0_2T_pred[:,0], dyal2_0_2T_pred[:,1], "--", color="red")
plt.scatter(dyal2_0_5T[:,0], dyal2_0_5T[:,1], facecolors="none", edgecolors="green")
plt.plot(dyal2_0_5T_pred[:,0], dyal2_0_5T_pred[:,1], "--", color="green")
plt.plot(T, -dS_2T*R/kB, ls="dotted", color="red")
plt.plot(T, -dS_5T*R/kB, ls="dotted", color="green")
plt.xlabel("T")
plt.ylabel("$-\Delta$S")
plt.legend()
plt.xlim(0, 140)
plt.hlines(0, 0, 140)
plt.savefig("dSvsT[001]_int.png", format="png", dpi=300, bbox_to_inches=True)

##############################################################################
# Heat Capacity calculations below

# Heat capacity array for different temperatures and fields
C = np.zeros((len(T), len(h)))

# Heat capacity calculation
for i in range(len(H)):
    
    C[:,i] = np.gradient(E[:,i], T)

# Plotting C vs T results

plt.scatter(dyal2_C[:,0], dyal2_C[:,1], facecolors="none", edgecolors="black")
plt.plot(dyal2_C_pred[:,0], dyal2_C_pred[:,1], ls="dashed", color="black")
plt.plot(dyal2_C_pred2[:,0], dyal2_C_pred2[:,1], ls="dotted", color="black")
plt.plot(T, C[:,0]*R/kB, color="black")
plt.xlabel("T")
plt.ylabel(r"Heat Capacity (J$\;\!$mol$^{-1}\;\!$K$^{-1}$)")
plt.xlim(0, 140)
plt.hlines(0, 0, 140)
plt.savefig("CvsT.png", format="png", dpi=300, bbox_to_inches=True)