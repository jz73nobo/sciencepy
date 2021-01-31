# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 14:39:20 2020

@author: Markus Meinert
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation


#----------------------------------------------------------------------#
#   Helper functions to display the MC snapshots
#----------------------------------------------------------------------#
def show_snapshots(snapshots, indexes=None):
    if indexes == None:
        indexes = list(range(snapshots.shape[0]))

    for i in indexes:
        plt.title('Step = %5i' % i)
        plt.imshow(snapshots[i], vmin=-1, vmax=1, interpolation='none', cmap='bwr')
        plt.show()

    return

def show_snapshots_array(snapshots, indexes=None):
    if indexes == None:
        indexes = list(range(snapshots.shape[0]))

    # find smallest square number larger than number of indexes
    L = len(indexes)
    S = L**0.5
    if int(S) == S:
        M = int(S)
    else:
        M = int(S) + 1
    
    i = 0
    fig, axs = plt.subplots(M, M, figsize=(5,5), dpi=200)
    for k in range(M):
        for l in range(M):
            axs[k, l].axis('off')
            if i < L:
                axs[k, l].imshow(snapshots[i], vmin=-1, vmax=1, interpolation='none', cmap='bwr')
            i += 1
    return

def anim_snapshots(snapshots, fps=30, dformat='mp4'):
    fig = plt.figure(figsize=(8,8))
    a = snapshots[0]
    im = plt.imshow(a, vmin=-1, vmax=1, interpolation='none', cmap='bwr', label='1')
    ax = fig.get_axes()
    ax[0].set_xlabel('Step = %5i' % 0)
    
    def animate_func(i):
        ax[0].set_xlabel('Step = %5i' % i)
        im.set_array(snapshots[i])
        return [im]
    
    anim = animation.FuncAnimation(
                                   fig,
                                   animate_func,
                                   frames=len(snapshots),
                                   interval=1000/fps,
                                   blit=True
                                   )
    if dformat == 'gif':
        anim.save('ising_anim.gif', writer=animation.PillowWriter(fps=fps))
    elif dformat == 'mp4':
        anim.save('ising_anim.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])
    else:
        raise ValueError('Data format not implemented: %s' % dformat)
  