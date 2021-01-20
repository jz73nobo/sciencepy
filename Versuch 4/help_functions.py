#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 08:57:33 2020

@author: Ion Gabriel Ion, Dimitrios Loukrezis
"""

import numpy as np
import scipy.sparse 
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import json 
import pickle

def load_from_file(fname):
    """
    Loads the grdig from the given file.

    Parameters
    ----------
    fname : string
        the name of the file.

    Returns
    -------
    pos : numpy array
        Contains the points of the grid. Has the shape (Np,2), where Np is the number of points.
        The first column are the x-coordinates and the second are the y-coordinates.
    conn : numpy array of integers
        Contains the connectivity matrix C as described in the script.
    bd : list of integers
        Contains the indices of the points which lie on the boundary.

    """
    with open(fname, 'rb') as file:
        data = pickle.load(file)
    pos = data["pos"]
    conn = data["conn"]
    bd = data["bd"]
    return pos,conn,bd


def save_to_file(fname,pos,conn,bd):
    """
    Save the geometry description to a file

    Parameters
    ----------
    fname : string
        the name of the file.
    pos : numpy array
        Contains the points of the grid. Has the shape (Np,2), where Np is the number of points.
        The first column are the x-coordinates and the second are the y-coordinates.
    conn : numpy array of integers
        Contains the connectivity matrix C as described in the script.
    bd : list of integers
        Contains the indices of the points which lie on the boundary.

    Returns
    -------
    None.

    """
    to_write = {'pos':pos, 'conn':conn, 'bd': bd}
    with open(fname, 'wb') as file:
        pickle.dump(to_write,file)

    
def analytical_eigenfrequencies_2d(Ne, c, lx, ly):
    """
    Computes the first Ne eigenvalues using the analytical solution

    Parameters
    ----------
    Ne : integer
         number of eigenvalues.
    c : float
        propagation speed.
    lx : float
         length in x direction.
    ly : float
         length in y direction.

    Returns
    -------
    ev : numpy 1d array
         the Ne eigenvalues
    """
    
    ##### Task 7.2 4)
    
   
    
    return None


def tensor_product_grid(ax,bx,ay,by,Nx,Ny):
    """
    constructs the grid corresponding to the box domain [ax,bx] x [ay,by].
    The number of 

    Parameters
    ----------
    ax : float
        lower bound in x-direction.
    bx : float
        upper bound in x-direction.
    ay : float
        lower bound in y-direction.
    by : float
        upper bound in y-direction.
    Nx : integer
        number of increments in x-direction.
    Ny : integer
        number of increments in y-direction.

    Returns
    -------
    pos : numpy array
        Contains the points of the grid. Has the shape (Np,2), where Np is the number of points.
        The first column are the x-coordinates and the second are the y-coordinates.
    conn : numpy array of integers
        Contains the connectivity matrix C as described in the script.
    bd : list of integers
        Contains the indices of the points which lie on the boundary.


    """
    
    ##### Task 7.1 3)
    xs = np.linspace(ax,bx,Nx)
    ys = np.linspace(ay,by,Ny)

    X,Y = np.meshgrid(xs,ys)
    IDX = np.arange(xs.size*ys.size).reshape(X.shape)

    Xm = np.hstack( (-np.ones((X.shape[0],1),dtype=np.int32),IDX[:,:-1]) )
    Xp = np.hstack( (IDX[:,1:],-np.ones((X.shape[0],1),dtype=np.int32)) )
    Ym = np.vstack( (-np.ones((1,X.shape[1]),dtype=np.int32),IDX[:-1,:]))
    Yp = np.vstack( (IDX[1:,:],-np.ones((1,X.shape[1]),dtype=np.int32)))

    pos = np.hstack((X.reshape([-1,1]),Y.reshape([-1,1])))
    conn = np.hstack((Xm.reshape([-1,1]),Xp.reshape([-1,1]),Ym.reshape([-1,1]),Yp.reshape([-1,1])))
    bd = list(set(IDX[:,0].tolist() + IDX[:,-1].tolist() + IDX[0,:].tolist() + IDX[-1,:].tolist() ))
    return pos,conn,bd
    


    
def construct_matrix(positions,connectivity,boundary):
    """
    Constructs the discrete Lapalce operator as explained in the script.

    Parameters
    ----------
    positions : numpy array
        Contains the points of the grid. Has the shape (Np,2), where Np is the number of points.
        The first column are the x-coordinates and the second are the y-coordinates.
    connectivity : numpy array of integers
        Contains the connectivity matrix C as described in the script.
    boundary : list of integers
        Contains the indices of the points which lie on the boundary.


    Returns
    -------
    L : scipy.sparse.coo_matrix
        The discrete Laplace operator as sparse matrix.

    """
    
    ##### Task 7.2 1)
   
    return None
    

def get_triangulation(positions,connectivity,boundary):
    """
    Compute the traingulation given the grid description.

    Parameters
    ----------
    positions : numpy array
        Contains the points of the grid. Has the shape (Np,2), where Np is the number of points.
        The first column are the x-coordinates and the second are the y-coordinates.
    connectivity : numpy array of integers
        Contains the connectivity matrix C as described in the script.
    boundary : list of integers
        Contains the indices of the points which lie on the boundary.


    Returns
    -------
    tri : numpy array
        Triangulation needed for tripcolor().

    """
    
    tri = []
    Np = positions.shape[0]
    for i in range(Np):
        if not i in boundary:
            tri.append([connectivity[i,2],i,connectivity[i,0]])
            tri.append([connectivity[i,1],i,connectivity[i,3]])
        else:
            if  connectivity[i,0]!=-1 and connectivity[i,2]!=-1 and (connectivity[i,1]==-1 or connectivity[i,3]==-1 or connectivity[i,1] in boundary or connectivity[i,3] in boundary):
                tri.append([connectivity[i,2],i,connectivity[i,0]])
            if  connectivity[i,1]!=-1 and connectivity[i,3]!=-1 and (connectivity[i,0]==-1 or connectivity[i,2]==-1 or connectivity[i,0] in boundary or connectivity[i,2] in boundary):
                tri.append([connectivity[i,1],i,connectivity[i,3]])
            
    tri = np.array(tri)
    return tri
    

def spectrum_signal(Gt, signal):
    """
    Perform the discrete Fourier transform of a given time signal

    Parameters
    ----------
    Gt : numpy 1d array
         time grid.
    signal : numpy 1d array
         time signal discretized on the given time grid.
    
         
    Returns
    -------
    freqs : numpy 1d array
            discrete frequencies.
    Cs    : numpy 1d array
            absolute values of the corresoponding Fourier-transformed values.
    """

    # only signals with an even size
    Gt = Gt[:Gt.size//2*2]
    signal = signal[:signal.size//2*2]
    
    # compute timestep and frequencies
    dt    = Gt[1] - Gt[0]
    freqs = np.linspace(0.0, 1.0/(2*dt), Gt.size//2)
    
    # perform Fast Fourier Transform (FFT)
    Cs = np.fft.fft(signal)
    Cs = np.abs(Cs[:Cs.size//2])
    Cs /= Cs.size  
   
    
    return freqs, Cs

def solve_timedomain(L, c, Gt, p0, v0):
    """
    Solves the wave equation on the given time grid with the initial 
    conditions u(x,y,t=0) = p0(x,y) and u'(x,y,t=0) = v0(x,y).

    Parameters
    ----------
    L    : numpy 2d array
           discrete Laplace.
    c    : float
           propagation speed.
    Gt   : numpy 1d array
           time grid.
    p0   : numpy 1d array
           initial condition u(t=0).
    v0   : numpy 1d array 
           initial velocity u'(t=0).

    Returns
    -------
    solution : numpy (Nx*Ny)xNt array
               The solution over all time steps. The solution is given in
               matrix format.
    """
    
    # Task 7.3 5)
    
    Nt = None
    
    # time update
    for i in range(1, Nt):
        if i % 100 == 0:
            print('time step:', i)
        
    return None