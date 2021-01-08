# -*- coding: utf-8 -*-
"""
Simple demot for the use of Pyomo

Problem

    min_x  \sum_i x_i
    s.t. x_i >= 0
         x_i - x_i-1 >= 1

@author: Steinke, Bott
"""

import pyomo.environ as pyo
from pyomo.opt import SolverFactory

# Step 0: Create an instance of the model
model = pyo.ConcreteModel()

# Step 1: Define index sets
I = range(5)

# Step 2: Define the decision 
model.x = pyo.Var(I)

# Step 3: Define Objective
model.cost = pyo.Objective(expr=sum(model.x[i] for i in I), sense=pyo.minimize)

# Step 4: Constraints
model.src = pyo.ConstraintList()
for i in I:
    model.src.add(model.x[i] >= 0)
for i in range(1, len(I)):
    model.src.add(model.x[i] - model.x[i - 1] >= 1)
model.pprint()

results = SolverFactory('glpk').solve(model)
results.write()

for i in I:
    print(pyo.value(model.x[i]))
