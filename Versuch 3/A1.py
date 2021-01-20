# -*- coding: utf-8 -*-
"""
Task 1: Line Search
"""

import numpy as np
import matplotlib.pyplot as plt
from OptLib import linesearch

f = lambda x : (x-2)**2


# xt = np.linspace(0,4)
xt = np.linspace(0,4, 50)
yt = list(map(f,xt))
ls_res = linesearch(f)


# Plotting
fig, ax = plt.subplots()
ax.plot(xt, yt)
ax.plot(ls_res['all_ts'],list(map(f,ls_res['all_ts'])), 'x')
ax.set_xlabel('x')
ax.set_ylabel('f(x)') 
