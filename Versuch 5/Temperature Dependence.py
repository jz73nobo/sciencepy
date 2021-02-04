# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 00:47:54 2021

@author: y50
"""
import matplotlib.pyplot as plt
import numpy as np
from numba import jit

#global parameters
J = 6.5*1e-21
miu = 2.04*1e-23
kb = 1.380649*1e-23
B = 0
V0 = 1.17*1e-29

N = M = 200
MCSTEPS = 50#0


@jit
def build_system_uniform(N,M):
    return np.ones((N,M))

@jit
def energy(system,n,m,B):    
    neighbours = system[(n+1)%N,m]+system[(n-1)%N,m]+system[n,(m+1)%M]+system[n,(m-1)%M]
    E = -J*system[n,m]* neighbours - miu*system[n,m]*B
    return E

@jit
def totalenergy(system,B):
    E_tot = 0
    for i in range(N):
        for j in range(M):
            E_mn = energy(system,i,j,0)
            E_tot += 0.5*E_mn+miu*system[i,j]*B
    return E_tot

#Metropolis algorithm
@jit
def mcmove(system,T,B):
    n,m = np.random.randint(0,N-1),np.random.randint(0,M-1)
    E_diff = -2*energy(system,n,m,B)
    b = 1/(kb*T)
    if E_diff > 0:
        r = np.random.rand()
        if np.exp(-b*E_diff) > r:
            system[n,m] = -system[n,m]
    else:
        system[n,m] = -system[n,m]
    return system

@jit
def main(system,T,B):
    snapshot = np.zeros((MCSTEPS,N,M))
    energy = np.zeros((MCSTEPS))
    for i in range(MCSTEPS):
        for j in range(N*M):
            system = mcmove(system, T, B)
        snapshot[i] = system
        energy[i] = totalenergy(system, B)
    return system,snapshot,energy

#Task 4.7 Magnetization, specific heat, and magnetic susceptibility
@jit
def magnetization(system):
    return miu*system.mean()

@jit
def specific_heat(system,T,B):
    E = np.zeros((N,M))
    b = 1/(kb*T)
    for i in range(N):
        for j in range(M):
            E[i,j] = energy(system,i,j,B)
    heat =  np.var(E) * kb * b**2
    return heat

@jit
def susceptibility(system,T,B): 
    miu0 = 4*np.pi*1e-7
    b = 1/(kb*T)
    return np.var(system) * miu0 * b * miu**2

 
@jit
def run_temperatures(start,end,num,B):
    S = build_system_uniform(N,M)
    m = h = s = np.array([])
    t = np.linspace(start,end,num)
    for i in range(num):
        S = main(S,t[i],B)[0]
        m = np.append(m,magnetization(S)/miu)
        h = np.append(h,specific_heat(S,t[i],B))
        s = np.append(s,susceptibility(S,t[i],B)/V0)
    return m,h,s


#temperature range
t = np.linspace(25,1500,60)

#analytical solution to normalized magnetization
an_sol = []
sol = lambda t: (1-np.sinh(2*J/(kb*t))**-4)**0.125
Tc = 2*J/(kb*np.log(1+2**0.5))
k = 0
while t[k] < Tc:
    an_sol.append(sol(t[k])) 
    k += 1 
    
    
m,h,s = run_temperatures(25,1500,60,B)
#draw pictures
plt.figure()
ax1 = plt.subplot(131)
plt.plot(t, m, ls = "dotted",lw=2, color="b",label="Monte Carlo")
plt.plot(t[:k], an_sol, color="r",label="analytical solution")
plt.axvline(Tc,ls="--",color="g")
plt.xlabel("temperature(K)")
plt.ylabel("normalized magnetization")
plt.legend()

ax2 = plt.subplot(132)
plt.plot(t, h)
plt.axvline(Tc,ls="--",color="g")
plt.xlabel("temperature(K)")
plt.ylabel("specific heat per spin (J/K)")


ax3 = plt.subplot(133)
plt.plot(t, s)
plt.axvline(Tc,ls="--",color="g")
plt.xlabel("temperature(K)")
plt.ylabel("magnetic susceptibility")


