#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 17:44:58 2020

@author: joaop

Backend for mean field approximation for the heinsenberg model with crystal
field corrections in the point charge approximation applied to DyAl2.

It imports the experimental data and contains the necessary functions for the
actual calculations.
"""

# Import useful packages
import numpy as np # Deal with arrays and matrices
import matplotlib.pyplot as plt # Plotting
from math import atan2 #
from numba import njit # Numba decorator for pre compilation

##############################################################################

# Get data

k100 = np.genfromtxt(r"kohake_100_4K.txt", delimiter=",")
k110 = np.genfromtxt(r"kohake_110_4K.txt", delimiter=",")
k111 = np.genfromtxt(r"kohake_111_4K.txt", delimiter=",")
theta_B = np.genfromtxt(r"theta_B.txt", delimiter=",")

dyal2_0_2T = np.genfromtxt("DyAl2_0_2T.txt", delimiter=",")
dyal2_0_2T_pred = np.genfromtxt("DyAl2_0_2T_pred.txt", delimiter=",")
dyal2_0_5T = np.genfromtxt("DyAl2_0_5T.txt", delimiter=",")
dyal2_0_5T_pred = np.genfromtxt("DyAl2_0_5T_pred.txt", delimiter=",")

dyal2_C = np.genfromtxt("DyAl2_C_exp.txt", delimiter=",")
dyal2_C_pred = np.genfromtxt("DyAl2_C_pred.txt", delimiter=",")
dyal2_C_pred2 = np.genfromtxt("DyAl2_C_pred2.txt", delimiter=",")

##############################################################################

# Constantes para o Disprósio (J = 15/2)
F4 = 60.
F6 = 13860.
X = 0.3
W = -0.011 * 1e-3 * 1.602e-19
# B4 = -5.5e-5 * 1.602e-22
# B6 = -5.6e-7 * 1.602e-22
B4 = -6.5e-5 * 1.602e-22
B6 = -5.5e-7 * 1.602e-22

g_Dy = 4/3
Z_Dy = 66
l_Dy = 44 / 1.602e-22

# Constantes Físicas
muB = 9.274e-24
kB = 1.38e-23
mu0 = np.pi * 4e-7
R  = 8.31446

##############################################################################

# Rotation matrix to rotate a vector around vector u by an angle theta
@njit
def Rot(theta, u):
    
    t = theta
    
    if (theta == 0. or np.linalg.norm(u) == 0.):
        
        return np.eye(3)
    
    else:
        
        u /= np.linalg.norm(u)
        x, y, z = u
        
        return np.array([[np.cos(t)+x**2*(1-np.cos(t)), x*y*(1-np.cos(t))-z*np.sin(t), x*z*(1-np.cos(t))+y*np.sin(t)],
                         [x*y*(1-np.cos(t))+z*np.sin(t), np.cos(t)+y**2*(1-np.cos(t)), y*z*(1-np.cos(t))-x*np.sin(t)],
                         [x*z*(1-np.cos(t))-y*np.sin(t), y*z*(1-np.cos(t))+x*np.sin(t), np.cos(t)+z**2*(1-np.cos(t))]])
Rot(np.pi/2, np.array([1.,0,0])) # Run a first time to execute numba pre compilation

# arctan2 function for arrays
def arctan2(y, x):
    
    r = np.copy(y)
    
    for i in range(len(r)):
    
        r[i] = atan2(y[i], x[i])
    return r

# Generate |j,m> ket state
@njit
def am_vec(j, m):
    
    vec = np.zeros(np.int(2*j+1), dtype=np.complex128)
    
    vec[np.int(m+j)] = 1
    
    return vec
am_vec(2,2)

# J^2 operator
@njit
def J2(vec):
    
    vec_out = vec.astype(np.complex128)
    
    j = (np.float64(len(vec))-1)/2
    
    return vec_out * j*(j+1)
J2(am_vec(2,2))

# J_z operator
@njit
def Jz(vec):
    
    vec_out = vec.astype(np.complex128)
    
    j = (np.float64(len(vec))-1)/2
    
    m = np.linspace(-j, j, np.int(2*j+1))
    
    return m * vec_out
Jz(am_vec(2,2))

# J_+ operator
@njit
def Jp(vec):
    
    vec_out = np.zeros(len(vec), dtype=np.complex128)
    
    vec_out[1:] = vec[:-1]
    
    j = (np.float64(len(vec))-1)/2
    
    m = np.linspace(-j, j, np.int(2*j+1))
    
    for i in range(np.int(2*j)):
        
        vec_out[i+1] *= np.sqrt( j*(j+1) - m[i]*(m[i]+1) )
        
    return vec_out
Jp(am_vec(2,2))

# J_- operator
@njit
def Jm(vec):
    
    vec_out = np.zeros(len(vec), dtype=np.complex128)
    
    vec_out[:-1] = vec[1:]
    
    j = (np.float64(len(vec))-1)/2
    
    m = np.linspace(-j, j, np.int(2*j+1))
    
    for i in range(1, np.int(2*j+1)):
        
        vec_out[i-1] *= np.sqrt( j*(j+1) - m[i]*(m[i]-1) )
        
    return vec_out
Jm(am_vec(2,2))

# J_x operator
@njit
def Jx(vec):
    
    return ( Jp(vec) + Jm(vec) ) / 2
Jx(am_vec(2,2))

# J_y operator
@njit
def Jy(vec):
    
    return ( Jp(vec) - Jm(vec) ) / 2j
Jy(am_vec(2,2))

# O_40 Stevens operator
@njit
def O40(vec):
    
    vec_out = vec.astype(np.complex128)
    
    j = (np.float64(len(vec))-1)/2
    
    m = np.linspace(-j, j, np.int(2*j+1))
    
    vec_out *= 35*m**4 - (30*j*(j+1) -  25)*m**2 + 3*(j*(j+1))**2 - 6*j*(j+1)
    
    return vec_out
O40(am_vec(2,2))

# O_44 Stevens operator
@njit
def O44(vec):
    
    vec_out = vec.astype(np.complex128)
    
    vec_out = ( Jp(Jp(Jp(Jp(vec)))) + Jm(Jm(Jm(Jm(vec)))) ) / 2
    
    return vec_out
O44(am_vec(2,2))

# O_60 Stevens operator
@njit
def O60(vec):
    
    vec_out = vec.astype(np.complex128)
    
    j = (np.float64(len(vec))-1)/2
    
    m = np.linspace(-j, j, np.int(2*j+1))
    
    vec_out *= ( 231*m**6 - (315*j*(j+1)-735)*m**4 + (105*(j*(j+1))**2-525*j*(j+1)+294)*m**2
                 -5*(j*(j+1))**3+40*(j*(j+1))**2-60*j*(j+1) )
    
    return vec_out
O40(am_vec(2,2))

# O_64 Stevens operator
@njit
def O64(vec):
    
    vec_out = np.zeros(len(vec), dtype=np.complex128)
    
    vec_out += Jp(Jp(Jp(Jp( 11*Jz(Jz(vec)) - J2(vec) - 38*vec ))))
    vec_out += Jm(Jm(Jm(Jm( 11*Jz(Jz(vec)) - J2(vec) - 38*vec ))))
    
    vec_temp = Jp(Jp(Jp(Jp(vec)))) + Jm(Jm(Jm(Jm(vec))))
    
    vec_out += 11*Jz(Jz(vec_temp)) - J2(vec_temp) - 38*vec_temp
    
    return vec_out / 4
O64(am_vec(2,2))

# Hamiltonian operator on j angular momentum states basis
@njit
def Hamiltonian(H, M, j):
    # H: magnetic field vector (T)
    # M: magnetization vector (Bohr magneton units)
    # j: total angular momentum
    
    V = -g_Dy * muB * (l_Dy * M + H)
    
    m = np.linspace(-j, j, np.int(2*j+1))
    
    mat = np.zeros((len(m), len(m)), dtype=np.complex128)
    
    for a in range(len(m)):
        for b in range(len(m)):
            
            bra = np.eye(len(m), dtype=np.complex128)[a]
            ket = np.eye(len(m), dtype=np.complex128)[b]
            
            ket_ = V[0] * Jx(ket) + V[1] * Jy(ket) + V[2] * Jz(ket)
            ket_ += B4 * ( O40(ket) + 5*O44(ket) )
            ket_ += B6 * ( O60(ket) - 21*O64(ket) )
            
            mat[a,b] = np.sum(np.conj(bra) * ket_)
            
    return mat
M = np.array([1., 1., 1.])*muB
H = np.array([1., 1., 1.])
j = 15/2
T = 4.2
Hamiltonian(H, M, 2)
E, E_kets = np.linalg.eigh(Hamiltonian(H, M, 2)) # returns eigenvalues and eigenvectors

# Partition function from energy spectrum and temperature
@njit
def part_func(E, T):
    # E: energy spectrum
    # T: temperature
    
    return np.sum(np.exp(-E/(kB*T)))
part_func(E, 1)

# Internal Energy (J)
@njit
def U(E, T):
    # E: energy spectrum
    # T: temperature    
    
    return np.sum(E*np.exp(-E/(kB*T))) / part_func(E, T)
U(E, 1)

# Free Energy (J)
@njit
def F(E, T, M):
    # E: energy spectrum
    # T: temperature
    # M: average magnetization
    
    return -kB*T*np.log(part_func(E, T)) + l_Dy / 2 * np.sum(M**2)
F(E, 1, M)

# Entropy (J/K)
@njit
def S(E, T):
    # E: energy spectrum
    # T: temperature
    
    return kB*( np.log(part_func(E, T)) + U(E, T)/(kB*T) )
S(E, 1)

# Get magnetization (J/T)
@njit
def mag(H, M, j, T):
    # H: magnetic field
    # M: average magnetization
    # j: total angular momentum
    # T: temperature
    E, E_kets = np.linalg.eigh(Hamiltonian(H, M, j))
    
    Z = part_func(E, T)
    
    Mx = 0j
    My = 0j
    Mz = 0j
    
    for i in range(len(E)):
        
        Mx += np.sum(np.conj(E_kets[:,i]) * Jx(E_kets[:,i])) * np.exp(-E[i]/(kB*T))
        My += np.sum(np.conj(E_kets[:,i]) * Jy(E_kets[:,i])) * np.exp(-E[i]/(kB*T))
        Mz += np.sum(np.conj(E_kets[:,i]) * Jz(E_kets[:,i])) * np.exp(-E[i]/(kB*T))
        
    return np.array([np.real(Mx), np.real(My), np.real(Mz)], dtype=np.float64) / Z * g_Dy * muB
mag(H, M, 2, 1)

# Magnetization component in the direction of the field
@njit
def mh(H, M):
    # H: magnetic field
    # M: average magnetization
    
    return np.sum(np.conj(H) * M) / np.linalg.norm(H)
mh(H, M)

# Get magnetization (J/T) minimizing free energy
@njit
def sc_mag_min_F(H, j, T, prec, cut, v1, v2):
    # H: magnetic field vector
    # j: total angular momentum
    # T: temperature
    # prec: precision in final magnetization (in units of bohr magnetons)
    # cut: minimum distance to already calculated values to stop useless calculations
    # v1: magnetization initial guess, which will rotate up to v2's direction
    # v2: direction up to which to rotate v1 searching for the correct magnetization
    
    n = 20 # Number of initial guesses rotating v1 up to v2's direction
    theta_max = np.arccos(min(1., np.sum(np.conj(v1) * v2)/np.linalg.norm(v1)/np.linalg.norm(v2)))
    theta = np.linspace(0, theta_max, n)
    
    lat = np.zeros((n, 3), dtype=np.float64)
    
    M_ = np.zeros((n, 3), dtype=np.float64)
    F_ = np.zeros(n)
    
    rot_vec = np.cross(v1, v2)
    
    for i in range(n):
        lat[i] = np.dot(Rot(theta[i], rot_vec), v1)
        
    M_list = np.array([[0,0,0]], dtype=np.float64)
    M0 = np.zeros(3)
    M = mag(H, M0, j, T)
    
    while (np.linalg.norm(M-M0)/muB > prec):
        
        M0 = M
        M = mag(H, M0, j, T)
        M_list = np.append(M_list, M).reshape(len(M_list)+1,3)

    for i in range(n):
        
        M0 = lat[i]*muB
        M = mag(H, M0, j, T)
        l = len(M_list)
        
        while (np.linalg.norm(M-M0)/muB > prec):
            
            if (np.min(np.sqrt(np.sum((M_list[:l]-M)**2, axis=1)))/muB < cut):
                M = np.copy(M_list[np.where(np.sqrt(np.sum((M_list[:l]-M)**2, axis=1))/muB < cut)[0][0]])
                break
            
            M0 = M
            M = mag(H, M0, j, T)
            M_list = np.append(M_list, M0).reshape(len(M_list)+1,3)
        
        M_[i] = M
        E = np.linalg.eigh(Hamiltonian(H, M, j))[0]
        F_[i] = F(E, T, M)

    return M_[np.where(F_ == np.min(F_))[0][0]]
sc_mag_min_F(H, 2., 1., 0.1, 0.5, np.array([0,0,1.]), np.array([0,0,1.]))