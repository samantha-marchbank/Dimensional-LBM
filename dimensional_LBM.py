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

#LBM
c = dx/dt           #Lattice velocity
cs = c/np.sqrt(3) #Speed of sound
N = 9             #Number of velocity directions
cxs = np.array([0,1,0,-1,0,1,-1,-1,1])                     #Discretised x-values
cys = np.array([0,0,1,0,-1,1,1,-1,-1])                     #Discretised y-values
ws = np.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36])   #Lattice Weightings
Fws = np.array([0,1/3,1/3,1/3,1/3,1/12,1/12,1/12,1/12])    #Pseudopotential Weightings
Sws = Fws*cs**2                                            #Weightings for switch function
idxs = np.array([0,1,2,3,4,5,6,7,8])        #Indexes of lattice directions
inv_idxs = np.array([0,3,4,1,2,7,8,5,6])    #And their inverses

periodic_x = True
periodic_y = True   # Change to False when adding y inlet/outlet

def neighbour_index(i, shift, size, periodic):
    j = i + shift
    if periodic:
        return j % size
    return min(max(j, 0), size - 1)

def wrap(quant, cx, cy):
    Nx, Ny = quant.shape
    result = np.empty_like(quant)
    for x in range(Nx):
        xn = neighbour_index(x, cx, Nx, periodic_x)
        for y in range(Ny):
            yn = neighbour_index(y, cy, Ny, periodic_y)
            result[x, y] = quant[xn, yn]
    return result

#choices
eta = 0.119

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
fluid = ~solids

fluid_float = (~solids).astype(float)
fluid_neighbours = [wrap(fluid_float, cxs[n], cys[n])for n in range(N)]
adj_denom = np.zeros((Nx, Ny))
for n in range(N):
    adj_denom += Fws[n]*fluid_neighbours[n]
valid_wall = solids & (adj_denom > 0)

#Create switch array for calculating adsorptive force
solid_bin = solids.astype(int)  
switch_x = np.zeros((Nx,Ny))    #solid neighbours in any x directions
switch_y = np.zeros((Nx,Ny))    #and in any y
for n in range(N):                                           #special weightings are used (Sws),
    switch_x += Sws[n]*wrap(solid_bin,cxs[n],cys[n])*cxs[n]  #rolls binary array in direction n,
    switch_y += Sws[n]*wrap(solid_bin,cxs[n],cys[n])*cys[n]  #and adds neighbours at unrolled location
switch_x[solids] = 0       #zero on solids themselves, even with solid neighbours
switch_y[solids] = 0





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


#>---<Peng-Robinson EOS Constants>---<#
R = 8.314   #j/mol/K

Tc_H = 33.15           
Pc_H = 1293000         
mm_H = 2.01588e-3                          #converted molar mass
ace_H = -0.22                 #acentric factor
m_H = 0.37464+1.5422*ace_H-0.26992*ace_H**2     #interim value
alpha_H = (1+m_H*(1-np.sqrt(T/Tc_H)))**2        #interim value
A_H = 0.45724*(R**2)*(Tc_H**2)/Pc_H  
B_H = 0.0778*R*Tc_H/Pc_H    
shift_H = 0

Tc_W = 647.096
Pc_W = 22.064E6
mm_W = 18.015E-3                          #and for water
ace_W = 0.344                 #acentric factor
m_W = 0.37464+1.5422*ace_W-0.26992*ace_W**2     #interim value
alpha_W = (1+m_W*(1-np.sqrt(T/Tc_W)))**2        #interim value
A_W = 0.45724*(R**2)*(Tc_W**2)/Pc_W  
B_W = 0.0778*R*Tc_W/Pc_W    
shift_W = -0.00024

def EOS_PR(dens, mm, A, B, alpha, shift):
    dens_ = dens/(1-shift*dens)
    n = dens_/mm
    P = (n*R*T)/(1-B*n) - A*alpha*n**2/(1+2*B*n-B**2*n**2)
    return P
print(EOS_PR(rho_W,mm_W,A_W,B_W,alpha_W,shift_W)/1E6)
          

#MRT
M = np.array([[1,1,1,1,1,1,1,1,1],[-4*c**2,-c**2,-c**2,-c**2,-c**2,2*c**2,2*c**2,2*c**2,2*c**2],     #Transformation matrix M
             [4*c**4,-2*c**4,-2*c**4,-2*c**4,-2*c**4,c**4,c**4,c**4,c**4],[0,c,0,-c,0,c,-c,-c,c],    #Gram-Schmidt formulation, dimensional
             [0,-2*c**3,0,2*c**3,0,c**3,-c**3,-c**3,c**3],[0,0,c,0,-c,c,c,-c,-c],
             [0,0,-2*c**3,0,2*c**3,c**3,c**3,-c**3,-c**3],[0,c**2,-c**2,c**2,-c**2,0,0,0,0],      #--Eq.20--#
             [0,0,0,0,0,c**2,-c**2,c**2,-c**2]],dtype=np.float64)
M_inv = np.linalg.inv(M)                   #Inverse transformation matrix
tau_e = tau_eps = 0.8*dt                      #general energy relaxation time
tau_q = 1.1*dt
tau_j = tau_p = 1.0*dt
lambda_ = np.empty((Nx, Ny, 9), dtype=np.float64)
lambda_[:,:,0] = 1.0/tau_p
lambda_[:,:,1] = 1.0/tau_e
lambda_[:,:,2] = 1.0/tau_eps
lambda_[:,:,3] = 1.0/tau_j
lambda_[:,:,4] = 1.0/tau_q
lambda_[:,:,5] = 1.0/tau_j
lambda_[:,:,6] = 1.0/tau_q

lambda_H = lambda_.copy()   #h2
lambda_H[:,:,7] = 1/tau_H   
lambda_H[:,:,8] = 1/tau_H

lambda_W = lambda_.copy()   #h20
lambda_W[:,:,7] = 1/tau_W
lambda_W[:,:,8] = 1/tau_W











#Funcs
def pseudopotential(P, rho, G, phase):
    psi_root = 2*(P*(dt/dx)**2 - rho/3)/(G)
    result = np.sqrt(psi_root)
    return result

def collision(F, rho, u_mag, ux, uy, S, C, lambda_X):
    #Find non-equilibrium moments              
    Nx,Ny,Nd = F.shape
    F_1D = np.ascontiguousarray(F).reshape(Nx*Ny,Nd)   #reshape Nx*Ny*Nd as a 1D array
    m = (F_1D@M.T).reshape(Nx,Ny,Nd)   #Multiply by the transpose of M and put back in Nx*Ny*Nd
    
    #Create moment array
    meq = np.empty((Nx,Ny,9))            #9 concatenated arrays for moments of:  #derived from eq3 in czelusniak
    meq[:,:,0] = rho                     #density
    meq[:,:,1] = rho*(-2*c**2+3*c**2*u_mag)        #energy
    meq[:,:,2] = rho*(c**4-3*c**2*u_mag)         #energy square
    meq[:,:,3] = rho*ux                  #x momentum
    meq[:,:,4] = -c**2*rho*ux                 #x direction energy flux
    meq[:,:,5] = rho*uy                  #y momentum
    meq[:,:,6] = -c**2*rho*uy                 #y direction energy flux
    meq[:,:,7] = rho*(ux**2-uy**2)       #normal stress
    meq[:,:,8] = rho*ux*uy               #shear stress
    
    #Collide and move populations back into realspace
    m_ = m - dt*lambda_X*(m-meq) + dt*(1-dt*lambda_X/2)*S + C          #--Eq.22--#
    m_flat = np.ascontiguousarray(m_).reshape(Nx*Ny,Nd)    #Same process as above 
    F = (m_flat@M_inv.T).reshape(Nx,Ny,Nd)                #Move back into realspace
    return F

def forcing_arrays(mForcex, mForcey, Forcex, Forcey, ux, uy, psi):
    S = np.zeros((Nx,Ny,9))              #another concatenated set of arrays (for moment space)
    mol_weight = (mForcex**2+mForcey**2)*(12*eta*dt**2)/(psi**2+1E-7)   #Additional forcing from intermolecular interactions
    S[:,:,1] = 6*(ux*Forcex+uy*Forcey) + mol_weight/(tau_e-dt/2)
    S[:,:,2] = -6*c**2*(ux*Forcex+uy*Forcey) - c**2*mol_weight/(tau_eps-dt/2)  #fourth order moments require correection c**2
    S[:,:,3] = Forcex
    S[:,:,4] = -c**2*Forcex
    S[:,:,5] = Forcey
    S[:,:,6] = -c**2*Forcey
    S[:,:,7] = 2*(ux*Forcex -uy*Forcey)
    S[:,:,8] = uy*Forcex + ux*Forcey         
    return S

def sum_pseud(pseud):
    Nx, Ny = pseud.shape
    adj_pseud = pseud.copy()
    
    #Pseudopotential at wall nodes, Cycle through directions in a numba complaint way
    for x in range(Nx):
        for y in range(Ny):
            if valid_wall[x,y]:
                total = 0
                for n in range(N):
                    xn = neighbour_index(x, cxs[n], Nx, periodic_x)
                    yn = neighbour_index(y, cys[n], Ny, periodic_y)
                    #Ensure non-periodicity in Y
                    if yn<0:
                        yn=0
                    elif yn>=Ny:
                        yn=Ny-1

                    #And also in X
                    if xn<0:
                        xn=0
                    elif xn>=Nx:
                        xn=Nx-1
                        
                    if fluid[xn,yn]:
                        total += Fws[n]*pseud[xn,yn]      #Add surrounding pseudopotential in N direction to total
                adj_pseud[x,y] = total/adj_denom[x,y]

    #Sums of neighbouring pseudopotential
    ads_pseud_x = np.zeros_like(pseud)  #Sum of solid-fluid pseudopotential interactions
    ads_pseud_y = np.zeros_like(pseud)
    sum_pseud_x = np.zeros_like(pseud)  #Sum of fluid-fluid pseudopotential interactions
    sum_pseud_y = np.zeros_like(pseud)
    for x in range(Nx):
        for y in range(Ny):
            if not fluid[x,y]:  #dont bother operating on solid nodes
                continue

            psi = pseud[x,y]
            ads_pseud_x[x,y] = psi*psi*switch_x[x,y]   #Eq.22 (part of it)
            ads_pseud_y[x,y] = psi*psi*switch_y[x,y]
            tempsum_x = 0
            tempsum_y = 0
            for n in range(1,N):  #Dont bother with 0th direction
                xn = neighbour_index(x, cxs[n], Nx, periodic_x)
                yn = neighbour_index(y, cys[n], Ny, periodic_y)
                if yn<0:
                    yn=0
                elif yn>=Ny:  #again non-periodicity in X and Y
                    yn=Ny-1
                if xn<0:
                    xn=0
                elif xn>=Nx:
                    xn=Nx-1
                neighbour = adj_pseud[xn, yn]   #Adjacent node in N direction
                tempsum_x += Fws[n]*neighbour*cxs[n]
                tempsum_y += Fws[n]*neighbour*cys[n]
            sum_pseud_x[x,y] = tempsum_x
            sum_pseud_y[x,y] = tempsum_y
    return sum_pseud_x, sum_pseud_y, ads_pseud_x, ads_pseud_y






def main():
    return    
    








