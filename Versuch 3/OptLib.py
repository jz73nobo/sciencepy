# -*- coding: utf-8 -*-
"""
Simple Optimization Library
"""

import numpy as np


def linesearch(fun, x_0=0, t_0=1e-3):
    '''
    Searches for the minimum of a given function using interval-doubling

    Parameters
    ----------
    fun : executable function returning a float value
        the function the minimal value is searched for
    
    x_0 : float, optional
        starting point. The default is 0.
    t_0 : float, optional
        initial step size. The default is 1e-3.

    Returns
    -------
    dict
        contains topt (estimated minimum) and all_ts (list of visited values)

    '''

    ############ Task 1 ############
    t = [t_0]
    x = [x_0,x_0+t[0]]
    y = [fun(x[0]),fun(x[1])]
    while y[-1] <= y[-2]:
        t.append(2*t[-1])
        x.append(x[-1]+t[-1])
        y.append(fun(x[-1]))
    return {'topt': t[-2], 'all_ts': t}


def gradDescent(fun, x_0, nmax=100, theta1=1e-3, theta2=1e-6):
    '''
    Performes a basic graident descent using linesearch to optimize the stepsize

    Parameters
    ----------
    fun : callable function
        represents the function to be minimized. A call returns a dict with entry "ff"
        for the function value (one value) and "gg" for its gradient (vector of floats)
    x_0 : float
        starting point
    nmax : int, optional
        maximal number of steps. The default is 100.
    theta1 : float, optional
        step-size termination condition. The default is 1e-3.
    theta2 : float, optional
        step-gain termination condition. The default is 1e-6.

    Returns
    -------
    x : np.array
        position of the minima.
    xs : list of np.array
        intermediate points during optimization.

    '''
    f = lambda x: fun(x)['value']
    g = lambda x: fun(x)['gradient']
    #alpha = linesearch(f,x_0)['topt']
    alpha = theta1
    x = [x_0,x_0 - alpha*g(x_0)]
    y = [f(x[0]),f(x[1])]

    while alpha>= theta1 and y[-2]-y[-1]>=theta2 and len(y) <= nmax: 
        
        x_current = x[-1] - alpha*g(x[-1])
        x.append(x_current)
        y.append(f(x_current))

    ############ Task 2 ############

    return x[-2], x
