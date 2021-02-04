# -*- coding: utf-8 -*-
import numpy as np
from numba import jit
import matplotlib.pyplot as plt

#global parameters
J = 6.5*1e-21
miu = 2.04*1e-23
kb = 1.380649*1e-23

    
start = -30
end = 30
num = 60
T = 950
N = M = 200
MCSTEPS = 50 #100 #500 #1000

B = np.concatenate((np.linspace(start,end-1,num),np.linspace(end,start-1,num)))

@jit
def build_system_uniform(N,M):
    return np.ones((N,M))

@jit
def energy(system,n,m,B):    
    neighbours = system[(n+1)%N,m]+system[(n-1)%N,m]+system[n,(m+1)%M]+system[n,(m-1)%M]
    E = -J*system[n,m]* neighbours - miu*system[n,m]*B
    return E


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
    for i in range(MCSTEPS):
        for j in range(N*M):
            system = mcmove(system, T, B)
    return system

#Task 4.7 Magnetization, specific heat, and magnetic susceptibility
@jit
def magnetization(system):
    return miu*system.mean()

@jit
def run_field_loop(start, end, num, T):
    S = build_system_uniform(N,M)
    m = np.array([])

    for i in range(2*num):
        S = main(S,T,B[i])
        m = np.append(m,magnetization(S)/miu)
    return m

m = run_field_loop(start, end, num, T)

plt.figure()
plt.plot(B, m)
plt.axvline(0,ls="--",color="black")
plt.axhline(0,ls="--",color="black")
plt.xlabel("magnetic flux density (T)")
plt.ylabel("normalized magnetization")
