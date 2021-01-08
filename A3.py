# -*- coding: utf-8 -*-
"""
Task 3: Optimization of battery operation
"""

import numpy as np
import matplotlib.pyplot as plt
from OptLib import gradDescent
import pyomo.environ as pyo
from pyomo.opt import SolverFactory


# Define linear programming function and gradient
def gradOpt(n, CB, PPV, piG, dt):
    '''

	Parameters
	----------
	n : int
		number of timesteps
	CB : float
		maximum capacity of the battery.
	PPV : np.array()
		avalilable solar energy per hour.
	piG : np.array()
		electricity price per hour.
	dt : int
		size of a timestep.

	Returns
	-------
	EB : float
		hourly level of the battery storrage.
	PB : np.array()
		hourly powerinput to the battery.
	PG : np.array()
		hourly power sold to the grid.

	'''

    def f(x, A, b, c, ll):
        # linear opimization problem:
        ii = A @ x - b > 0
        ff = c.T @ x + ll * ii.T @ (A @ x - b)
        gg = c + ll * A.T @ ii
        return {'value': ff, 'gradient': gg}

    ############ Task 3 Add code here ############
    # define Matrices and vectors for linear optimization
    A = np.eye(n)
    b = CB
    c = piG

    ff = lambda x: f(x, A, b, c, 1e2)

    # perform optimization
    ##### TASK 3: Add code here ##########
    x, xs = gradDescent(ff, 0., n, theta1=1e-3, theta2=1e-6)
    x1, xs1 = gradDescent(ff, 0., n-1, theta1=1e-3, theta2=1e-6)
    # calculate all values:
    ##### TASK 3: Add code here ##########

    EB = np.vstack(xs)
    PB = (np.vstack(xs)-np.vstack(xs1))/dt
    PG = PB + PPV

    return EB, PB, PG


def pyomoOpt(n, CB, PPV, piG, dt):
    '''

	Parameters
	----------
	n : int
		number of timesteps. 
	CB : float
		maximum capacity of the battery.
	PPV : np.array()
		avalilable solar energy per hour.
	piG : np.array()
		electricity price per hour.
	dt : int
		size of a timestep.

	Returns
	-------
	EB : float
		hourly level of the battery storrage.
	PB : np.array()
		hourly powerinput to the battery.
	PG : np.array()
		hourly power sold to the grid.

	'''
    ############ Task 4 ############
    return np.zeros(n), np.zeros(n), np.zeros(n)


# define problem parameters
n = 24
t = np.linspace(1, 24, n)
PPV = np.sin(t / 24 * np.pi)  # available solar energy
piG = 5 + 3 * np.cos((t + 4) / 24 * np.pi * 2)  # assumed wholesale price level in ct per kWh
dt = 24 / n
CB = 3

# overwrite parameters for simple debug case
debugMode = False
if debugMode:
    n = 2
    t = range(n)
    PPV = np.array([10, 0])
    piG = np.array([0, 5])
    dt = 1

# do the optimization
EB, PB, PG = gradOpt(n, CB, PPV, piG, dt)
# EB, PB, PG = pyomoOpt(n, CB, PPV, piG, dt)


# %% Plotting

# Plot Battery test

# Area plot of final operation 
fig, axs = plt.subplots(3, 1)
axs[0].plot(t, PPV)
axs[0].plot(t, PG)
axs[0].plot(t, PB)
axs[0].set_title('Power (kW)')
axs[0].legend(['PV', 'PG', 'PB'])

# Function values during optimization
axs[1].plot(t, piG)
axs[1].set_title('Buying Price (ct/kWh)')

# Function values during optimization
axs[2].plot(t, EB)
axs[2].set_title('Battery level (kWh)')

plt.tight_layout()
