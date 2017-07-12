#GOL2D_partition_helper.py
"""
This little bit of hackery is where I will save functions for mpi4py version of 
Conway's Game of Life.  I am not implementing it; merely trying to help
overcome obstacles for the one who is.

"""
from pymetis import part_graph
import numpy as np


ex = [1, 0, -1 ,0, 1, -1, -1, 1]
ey = [0, 1, 0, -1, 1, 1, -1, -1]

def getPartitionAndOffset(partition,partNum):
    """
    partition - a numpy array listing the partition number for all points
    partNum - an integer indicating which partition is calling this function

    return: a numpy array of integers giving the points in the partition ordered by global node number
            an integer offset for writing MPI files
    """

    partList = []
    intOffset = 0
    num = 0 # global node number
    for p in np.nditer(partition):
        if p<partNum:
            intOffset+=1
        if p==partNum:
            partList.append(num)
        num+=1 #increment the global node number

    return np.array(partList).astype('int32'), intOffset

def set_adjacency(Nx,Ny,ex,ey):
    """
    Nx = num of board positions in X-direction
    Ny = num of board positions in Y-direction
    ex = x-component of direction to neighbors
    ey = y-component of direction to neighbors
    
    
    returns adjDict = dictionary where the keys are the global lattice point numbers
    and the values are lists of neighboring lattice points
    
    
    """
    adjDict = {}
    
    for y in range(Ny):
      for x in range(Nx):
        gid = x + y*Nx
        for spd in range(len(ex)):
           dx = int(ex[spd]); dy = int(ey[spd]); 
           tx = (x+dx)%Nx; ty= (y+dy)%Ny; # target x- and y- cell numbers
           tid = tx+ty*Nx; # target point global id
           adjDict.setdefault(gid,[]).append(tid)
                    
    return adjDict

def makePartition(Nx,Ny,numParts):
    """
    Nx = number of board positions in the X-direction
    Ny = number of board positions in the Y-direction
    numParts = the number of partitions to make

    returns partList = list of which partitions each board position will be in
    """
    adjDict = set_adjacency(Nx,Ny,ex,ey);
    cuts,partList = part_graph(numParts,adjDict);
    return partList


