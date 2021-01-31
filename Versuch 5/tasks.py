# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 09:27:23 2021

@author: y50
"""
import matplotlib.pyplot as plt
import numpy as np
from help_functions import *

#global parameters
J = 6.5*1e-21
miu = 2.04*1e-23
kb = 1.380649*1e-23
T = 500
B = 0

N = M = 21
MCSTEPS = 100
#Task 4.1 Setting up the Ising Lattice
def build_system_uniform(N,M):
    return np.ones((N,M))

def build_system_random(N,M):
    ans = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            ans[i,j] = np.random.choice([-1,1])
    return ans

Mu = build_system_uniform(21,21)
Mr = build_system_random(21,21)
plt.matshow(Mr)
plt.show()

#Task 4.2 Calculating the energy of one spin
def energy(system,n,m,B):    
    neighbours = 0
    if n < N-1:
        neighbours += system[n+1,m]
    if n > 0:
        neighbours += system[n-1,m]
    if m < M-1:
        neighbours += system[n,m+1]
    if m > 0:
        neighbours += system[n,m-1]
    E = -J*system[n,m]* neighbours - miu*system[n,m]*B
    return E
#Task 4.3 Calculating the total energy
def totalenergy(system,B):
    E_tot = 0
    for i in range(N):
        for j in range(M):
            E_mn = energy(system,i,j,0)
            E_tot += 0.5*E_mn+miu*system[i,j]*B
    return E_tot

#Verify the implementation by checking that E_tot = -2*N*M*J when B=0
E = totalenergy(Mu, B)
diff = abs(-2*N*M*J-E)
print('difference between E_tot and analytical solution:',diff)

#Task 4.4 Metropolis algorithm
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
def main(system,T,B):
    snapshot = np.zeros((MCSTEPS,N,M))
    energy = np.zeros((MCSTEPS))
    for i in range(MCSTEPS):
        for j in range(N*M):
            system = mcmove(system, T, B)
        snapshot[i] = system
        energy[i] = totalenergy(system, B)
    return system, snapshot, energy
N = M = 50
s = build_system_random(N,M)
s,snapshot,E_tot = main(s, T, B)
show_snapshots_array(snapshot)
