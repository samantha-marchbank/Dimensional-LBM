# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 11:18:18 2026

@author: s2267401
"""
import numpy as np
import matplotlib.pyplot as plt


#####Setup
Lx = 0.03  #m
Ly = 0.2  #m
Lwidth = 0.01  #width of plates
Llength = 0.09  #length of plates
dx = 0.001 #m
dt = 1 #s

Nx = int(Lx/dx)
Ny = int(Ly/dx)   #dx=dy
Nwidth = int(Lwidth/dx)
Nlength = int(Llength/dx)
Nbuffer = int((Ny-Nlength)/2)

#Stenciling
Total = np.zeros((Nx,Ny))
solids =  np.zeros(Total.shape,dtype=bool)
solids[:Nwidth,Nbuffer:Nbuffer+Nlength] = True
solids[Nx-Nwidth:,Nbuffer:Nbuffer+Nlength] = True

#LBM
c = dx/dt           #Lattice velocity
cs = c/np.sqrt(1/3) #Speed of sound
N = 9             #Number of velocity directions
cxs = np.array([0,1,0,-1,0,1,-1,-1,1])                     #Discretised x-values
cys = np.array([0,0,1,0,-1,1,1,-1,-1])                     #Discretised y-values
ws = np.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36])   #Lattice Weightings
Fws = np.array([0,1/3,1/3,1/3,1/3,1/12,1/12,1/12,1/12])    #Pseudopotential Weightings
Sws = Fws*cs**2                                            #Weightings for switch function
idxs = np.array([0,1,2,3,4,5,6,7,8])        #Indexes of lattice directions
inv_idxs = np.array([0,3,4,1,2,7,8,5,6])    #And their inverses

#Fluid properties
T = 323 #K
P = 10E6 #Pa
rho_W = 992.2 #kg/m3
rho_H = 7.1 #kg/m3
Dvisc_H = 9.63E-6 #Pa s
Dvisc_W = 5.51E-4 #Pa s
Kvisc_H = Dvisc_H/rho_H
Kvisc_W = Dvisc_W/rho_W
tau_H = Kvisc_H/cs**2+0.5*dt
tau_W = Kvisc_W/cs**2+0.5*dt


#Funcs
def pseudopotential(P, rho, G, phase):
    psi_root = 2*(P - rho*cs**2)/(G*c**2)
    result = np.sqrt(psi_root)
    return result

def collision(F, M, rho, u_mag, ux, uy, tau, S, C, G):
    #Find non-equilibrium moments              
    Nx,Ny,Nd = F.shape
    F_1D = np.ascontiguousarray(F).reshape(Nx*Ny,Nd)   #reshape Nx*Ny*Nd as a 1D array
    m = (F_1D@M.T).reshape(Nx,Ny,Nd)   #Multiply by the transpose of M and put back in Nx*Ny*Nd
    
    #Create moment array
    meq = np.empty((Nx,Ny,9))            #9 concatenated arrays for moments of:
    meq[:,:,0] = rho                     #density
    meq[:,:,1] = rho*(-2+3*u_mag)        #energy
    meq[:,:,2] = rho*(1-3*u_mag)         #energy square
    meq[:,:,3] = rho*ux                  #x momentum
    meq[:,:,4] = -rho*ux                 #x direction energy flux
    meq[:,:,5] = rho*uy                  #y momentum
    meq[:,:,6] = -rho*uy                 #y direction energy flux
    meq[:,:,7] = rho*(ux**2-uy**2)       #normal stress
    meq[:,:,8] = rho*ux*uy               #shear stress
    
    #Calculate relaxation time and diagonal relaxation matrix
    lambda_ = np.empty((Nx, Ny, 9), dtype=np.float64)
    lambda_[:, :, 0] = 1.0
    lambda_[:, :, 1] = 1.0 / 0.8
    lambda_[:, :, 2] = 1.0 / 0.8
    lambda_[:, :, 3] = 1.0
    lambda_[:, :, 4] = 1.0 / 1.1
    lambda_[:, :, 5] = 1.0
    lambda_[:, :, 6] = 1.0 / 1.1
    lambda_[:, :, 7] = 1.0 / tau
    lambda_[:, :, 8] = 1.0 / tau
    
    #Collide and move populations back into realspace
    m_ = m - lambda_*(m-meq) + (1-lambda_/2)*S + C          #--Eq.22--#
    m_flat = np.ascontiguousarray(m_).reshape(Nx*Ny,Nd)    #Same process as above 
    F = (m_flat@M_inv.T).reshape(Nx,Ny,Nd)                #Move back into realspace
    return F



plt.imshow(solids)