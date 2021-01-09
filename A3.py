# -*- coding: utf-8 -*-
"""
Task 3: Optimization of battery operation
"""

import numpy as np
import matplotlib.pyplot as plt
from OptLib import gradDescent
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from sympy import diff

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
    A = np.linspace(1, 24, n)
    b = CB
    c = np.zeros(n)

    ff = lambda x: f(x, A, b, c, 1e2)

    # perform optimization
    ##### TASK 3: Add code here ##########
    x, xs = gradDescent(ff, 0., n, theta1=1e-3, theta2=1e-6)
    # calculate all values:
    ##### TASK 3: Add code here ##########

    EB = x
    PB = diff(ff,EB)
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
    # Step 0: Create an instance of the model
    model = pyo.ConcreteModel()
    
    # Step 1: Define index sets
    I = range(n)
    
    # Step 2: Define the decision 
    model.x = pyo.Var(I)
    
    # Step 3: Define Objective
    model.cost = pyo.Objective(expr=sum(piG @ model.x[t] for t in I), sense=pyo.maximize)
    
    # Step 4: Constraints
    model.src = pyo.ConstraintList()
    EB = []
    PB = []
    PG = []
    for t in I:
        model.src.add((model.x[t] >= 0)&(model.x[t]<=CB))
    for t in range(1, len(I)):
        model.src.add((model.x[t] - model.x[t - 1])/dt + PPV >= 0)
    model.pprint()
    
    results = SolverFactory('glpk').solve(model)
    results.write()
    
    for t in I:
        EB.append(pyo.value(model.x[t]))
        PB.append(pyo.value((model.x[t])-pyo.value(model.x[t-dt]))/dt)
    PG = EB + PB
    return EB, PB, PG


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
# EB, PB, PG = gradOpt(n, CB, PPV, piG, dt)
EB, PB, PG = pyomoOpt(n, CB, PPV, piG, dt)


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
