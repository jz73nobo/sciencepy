# -*- coding: utf-8 -*-
"""
Task 2: Gradient Descent
"""

import numpy as np
import matplotlib.pyplot as plt
from OptLib import gradDescent


def f(x):
    ff = np.cos(x[0]) + np.cos(x[1]) - sum(x) / 10
    gg = np.array([-np.sin(x[0]) - 1 / 10, -np.sin(x[1]) - 1 / 10])
    return {'value': ff, 'gradient': gg}


def g(x, phi, d1, d2, mu):
    '''
	function

	Parameters
	----------
	x : 2d-np.array
		point of evaluation.
	phi : float between 0 and 2pi
		rotation angle.
	d1 : float
		stretching factor in x.
	d2 : float
		stretching factor in y.
	mu : 2d-np.array
		shiftingvector of the center

	Returns
	-------
	dict
		contains the 'value' and the 'gradient' of the function

	'''

    R = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])  # rotation matrix
    D = np.diag([d1, d2])  # streching matrix
    A = R.T @ D @ R
    b = -2 * A @ mu

    # function value:
    ff = x.T @ A @ x + b.T @ x + mu.T @ A @ mu
    # gradient
    gg = 2 * A.T @ x + b
    return {'value': ff, 'gradient': gg}


# select the right function
fun = f
# fun = lambda x : g(x, phi=-np.pi/3, d1=1, d2=5, mu=np.array([3,5]))


# perform gradient descent
x0 = np.array([7, 0])  # starting point
xopt, xs = gradDescent(fun, x0, nmax=100)

# generate grid for vizualization
hh = np.linspace(-1, 12)
x1t, x2t = np.meshgrid(hh, hh)
yt = np.zeros((len(hh), len(hh)))
for i in range(len(hh)):
    for j in range(len(hh)):
        yt[i, j] = fun(np.array([x1t[i, j], x2t[i, j]]))['value']

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2)
cf = ax1.contourf(x1t, x2t, yt)
for x in xs:
    ax1.plot(x[0], x[1], 'kx')
    gg = fun(x)['gradient']
    gg = gg / max(np.linalg.norm(gg), 1e-6)
    ax1.arrow(x[0], x[1], gg[0], gg[1], color='r')
fig.colorbar(cf, ax=ax1, shrink=0.9)
ax1.set_title('Opt. function and visited positions incl. gradient direction')
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')

ax2.plot(list(map(lambda x: fun(x)['value'], xs)))
ax2.set_title('Obtained function values during gradient descent')
ax2.set_xlabel('Iteration number')
ax2.set_ylabel('Fucntion values')
