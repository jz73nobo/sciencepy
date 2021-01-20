#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 14:42:13 2020

@author: Ion Gabriel Ion, Dimitrios Loukrezis
"""

import numpy as np
import scipy.sparse 
import scipy.sparse.linalg
import matplotlib.pyplot as plt


from help_functions import *

##### Task 7.1 1)
# read from file 
filename = 'Lshape_1.txt' 
#filename = 'Lshape_2.txt'
#filename = 'Cshape_1.txt'
#filename = 'Cshape_2.txt'
pt,C,bd = load_from_file(filename)

x = pt[:,0]
y = pt[:,1]

x_bd,y_bd = [],[]
for _ in bd:
    x_bd.append(x[_])
    y_bd.append(y[_])
    
plt.figure()
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(x,y,c='#0000FF')
plt.scatter(x_bd,y_bd,c='#FF0000')
plt.title(filename)
#####

##### Task 7.1 2)
f = lambda x,y: np.sin(2*np.pi*x)*np.sin(2*np.pi*y)

tri = get_triangulation(pt,C,bd)
center = np.sum(pt[tri[:,:2]], axis=1)/3.0
color = np.array([f(x,y) for x, y in center])


fig,ax1=plt.subplots()
ax1.set_aspect('equal')

tpc = ax1.tripcolor(x,y,tri,facecolors=color,edgecolors='k')
fig.colorbar(tpc)


#####
