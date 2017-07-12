#!/usr/bin/env python
"""
demo program for 2D Game of Life implementation

"""

from mpi4py import MPI
import numpy as np
from copy import copy, deepcopy
import random
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
import sys
sys.path.insert(1,'.') #<-- required for loading local modules
from vtkHelper import saveStructuredPointsVTK_ascii as writeVTK
import GOL2D_partition_helper as gp

#border point where the values are the global location of the point and
#the rank that requires the point
class bPoint:
  def __init__(self,rank,point):
    self.rank = rank
    self.point = point
  def __str__(self):
    return str(self.__dict__)
  def __eq__(self,other): #<-allows for "==" operator to be used
    return self.__dict__ == other.__dict__

#symbolizes a data point to be sent to another process
class dPoint:
  def __init__(self,data,point):
    self.data = data
    self.point = point
  def __str__(self):
    return str(self.__dict__)
  def __eq__(self,other): #<-allows for "==" operator to be used
    return self.__dict__ == other.__dict__
    
#adds a point to a list if the point is not already in said list
def addP(n,list):
  if n in list:
    "Do Nothing"
  else:
    list.append(n) 

#creates a list of type bPoint for the calling rank
def myBorder(partVert,numPart,Nx,rank):
  border = []
  for i in range(len(partVert)):
    if partVert[i] == rank:
      #right
      if partVert[i] != partVert[(i+1+len(partVert))%len(partVert)]:
        n = bPoint(partVert[(i+1+len(partVert))%len(partVert)],i)
        addP(n,border)
      #left
      if partVert[i] != partVert[(i-1+len(partVert))%len(partVert)]:
        n = bPoint(partVert[(i-1+len(partVert))%len(partVert)],i)
        addP(n,border)
      #down
      if partVert[i] != partVert[(i-Nx+len(partVert))%len(partVert)]:
        n = bPoint(partVert[(i-Nx+len(partVert))%len(partVert)],i)
        addP(n,border)
      #up
      if partVert[i] != partVert[(i+Nx+len(partVert))%len(partVert)]:
        n = bPoint(partVert[(i+Nx+len(partVert))%len(partVert)],i)
        addP(n,border)
      #diagonals
      if partVert[i] != partVert[(i-Nx+1+len(partVert))%len(partVert)]:
        n = bPoint(partVert[(i-Nx+1+len(partVert))%len(partVert)],i)
        addP(n,border)
      if partVert[i] != partVert[(i-Nx-1+len(partVert))%len(partVert)]:
        n = bPoint(partVert[(i-Nx-1+len(partVert))%len(partVert)],i)
        addP(n,border)
      if partVert[i] != partVert[(i+Nx+1+len(partVert))%len(partVert)]:
        n = bPoint(partVert[(i+Nx+1+len(partVert))%len(partVert)],i)
        addP(n,border)
      if partVert[i] != partVert[(i+Nx-1+len(partVert))%len(partVert)]:
        n = bPoint(partVert[(i+Nx-1+len(partVert))%len(partVert)],i)
        addP(n,border)
  return border

#return dictionary of neighbors
def findN(myBorder):
  myNeighbors = {}
  n = []
  for i in range(len(myBorder)):
    addP(myBorder[i].rank,n)
  for i in range(len(n)):
    myNeighbors[i] = n[i]
  inv_n = {v: k for k, v in myNeighbors.items()}
  return myNeighbors, inv_n
    

#records the inner points that require no outside data
def findInner(myPart,myBorder):
  inner = []
  for p in range(len(myPart)):
    inn = 0
    for j in range(len(myBorder)):
      if myBorder[j].point == myPart[p]:
        inn += 1
      else:
        "Do nothing"
    if inn < 1:
      inner.append(myPart[p])
  return inner

#creates an array of "cells" that make up the partition 0-dead 1-alive
def makeData(myPart):
  myData = [0 for i in range(len(myPart))]
  for i in range(len(myData)):
    myData[i] = dPoint(random.randint(0,1),myPart[i])
  return myData

#the own partition, the inner, two copies of data so the states do not effect each other
def calcInner(myPart,myInner,myData1,myData2,Nx):
  for i in range(len(myInner)):
    #right
    j = np.where(myPart==((myInner[i]+1+(Nx*Nx))%(Nx*Nx)))
    nTot = myData2[(j[0][0])].data
    #left
    j = np.where(myPart==((myInner[i]-1+(Nx*Nx))%(Nx*Nx)))
    nTot += myData2[(j[0][0])].data
    #down
    j = np.where(myPart==((myInner[i]-Nx+(Nx*Nx))%(Nx*Nx)))
    nTot += myData2[(j[0][0])].data
    #up
    j = np.where(myPart==((myInner[i]+Nx+(Nx*Nx))%(Nx*Nx)))
    nTot += myData2[(j[0][0])].data
    #diagonals
    j = np.where(myPart==((myInner[i]-Nx+1+(Nx*Nx))%(Nx*Nx)))
    nTot += myData2[(j[0][0])].data
    j = np.where(myPart==((myInner[i]-Nx-1+(Nx*Nx))%(Nx*Nx)))
    nTot += myData2[(j[0][0])].data
    j = np.where(myPart==((myInner[i]+Nx+1+(Nx*Nx))%(Nx*Nx)))
    nTot += myData2[(j[0][0])].data
    j = np.where(myPart==((myInner[i]+Nx-1+(Nx*Nx))%(Nx*Nx)))
    nTot += myData2[(j[0][0])].data
    j = np.where(myPart==(myInner[i]))
    if myData2[(j[0][0])].data == 1:
      # fewer than two neighbors or more than three dies
      # two or three lives
      if nTot < 2 or nTot > 3:
        myData1[(j[0][0])].data = 0
      else:
        myData1[(j[0][0])].data = 1
    else:
      # three neighbors comes alive
      if nTot == 3:
        myData1[(j[0][0])].data = 1
      else:
        myData1[(j[0][0])].data = 0

#creates the initial sending buffer
def sendIt(myData,myBorder,myNeighbors,inv_n):
  send = [[0 for i in range(0)] for j in range(len(myNeighbors))]
  for i in range(len(myBorder)):
    for j in range(len(myData)):
      if myData[j].point == myBorder[i].point:
        send[inv_n[myBorder[i].rank]].append(myData[j])
  return send

#updates the sending buffer to match the current state
def updateSend(myData, send):
  for i in range(len(send)):
    for j in range(len(send[i])):
      for x in range(len(myData)):
        if myData[x].point == send[i][j].point:
          send[i][j] = myData[x]

#Really ugly calculation of the border
def calcBorder(myPart,myBorder,get,myData1,myData2,Nx):
  for i in range(len(myBorder)):
    nTot = 0
    n = 0
    for j in range(len(myPart)):
      if nTot == 4 or n == 8:
        break
      if myPart[j]==((myBorder[i].point+1+(Nx*Nx))%(Nx*Nx)):
        nTot += myData2[j].data
        n += 1
        if nTot == 4 or n == 8:
          break
      if myPart[j]==((myBorder[i].point-1+(Nx*Nx))%(Nx*Nx)):
        nTot += myData2[j].data
        n += 1
        if nTot == 4 or n == 8:
          break
      if myPart[j]==((myBorder[i].point-Nx+(Nx*Nx))%(Nx*Nx)):
        nTot += myData2[j].data
        n += 1
        if nTot == 4 or n == 8:
          break
      if myPart[j]==((myBorder[i].point+Nx+(Nx*Nx))%(Nx*Nx)):
        nTot += myData2[j].data
        n += 1
        if nTot == 4 or n == 8:
          break
      if myPart[j]==((myBorder[i].point+1-Nx+(Nx*Nx))%(Nx*Nx)):
        nTot += myData2[j].data
        n += 1
        if nTot == 4 or n == 8:
          break
      if myPart[j]==((myBorder[i].point-1+Nx+(Nx*Nx))%(Nx*Nx)):
        nTot += myData2[j].data
        if nTot == 4:
          break
      if myPart[j]==((myBorder[i].point-1-Nx+(Nx*Nx))%(Nx*Nx)):
        nTot += myData2[j].data
        n += 1
        if nTot == 4 or n == 8:
          break
      if myPart[j]==((myBorder[i].point+1+Nx+(Nx*Nx))%(Nx*Nx)):
        nTot += myData2[j].data
        n += 1
        if nTot == 4 or n == 8:
          break
    for x in range(len(get)):
      for y in range(len(get[x])):
        if nTot == 4 or n == 8:
          break
        if get[x][y].point==((myBorder[i].point+1+(Nx*Nx))%(Nx*Nx)):
          nTot += get[x][y].data
          n += 1
          if nTot == 4 or n == 8:
            break
        if get[x][y].point==((myBorder[i].point-1+(Nx*Nx))%(Nx*Nx)):
          nTot += get[x][y].data
          n += 1
          if nTot == 4 or n == 8:
            break
        if get[x][y].point==((myBorder[i].point+Nx+(Nx*Nx))%(Nx*Nx)):
          nTot += get[x][y].data
          n += 1
          if nTot == 4 or n == 8:
            break
        if get[x][y].point==((myBorder[i].point-Nx+(Nx*Nx))%(Nx*Nx)):
          nTot += get[x][y].data
          n += 1
          if nTot == 4 or n == 8:
            break
        if get[x][y].point==((myBorder[i].point+1+Nx+(Nx*Nx))%(Nx*Nx)):
          nTot += get[x][y].data
          n += 1
          if nTot == 4 or n == 8:
            break
        if get[x][y].point==((myBorder[i].point-1+Nx+(Nx*Nx))%(Nx*Nx)):
          nTot += get[x][y].data
          n += 1
          if nTot == 4 or n == 8:
            break
        if get[x][y].point==((myBorder[i].point+1-Nx+(Nx*Nx))%(Nx*Nx)):
          nTot += get[x][y].data
          n += 1
          if nTot == 4 or n == 8:
            break
        if get[x][y].point==((myBorder[i].point-1-Nx+(Nx*Nx))%(Nx*Nx)):
          nTot += get[x][y].data
          n += 1
          if nTot == 4 or n == 8:
            break

    for j in range(len(myPart)):
      if myPart[j]==myBorder[i].point:
        if myData2[j].data == 1:
        # fewer than two neighbors or more than three dies
        # two or three lives
          if nTot < 2 or nTot > 3:
            myData1[j].data = 0
          else:
            myData1[j].data = 1
        else:
          # three neighbors comes alive
          if nTot == 3:
            myData1[j].data = 1
          else:
            myData1[j].data = 0

def writeBuffer(buffer,myData):
  for i in range(len(buffer)):
    buffer[i] = myData[i].data
      
       
#uncomment when ready to test performance
start_time = time.time()

Nx = 90;
Ny = 90;
numPart = size

# rank 0 process do the partitioning.
# save the result to a binary data file "partition.gol"
# all processes will read this file

partFileName = "partition.gol"

if rank==0:
    partVert = gp.makePartition(Nx,Ny,numPart);
    #print "partVert = ",partVert #<--- printed as a Python list
    # WORKS: THE GLBOAL LOCATIONS ARE LIKE A GRAPH 0 IS BOTTOM LEFT
    np.array(partVert).astype('int32').tofile(partFileName) #saved as a numpy array
   
comm.Barrier() #everyone waits until rank 0 is done

# everyone reads the file
partVert = np.fromfile(partFileName,dtype='int32') #<-- loaded as a numpy array
# now all processes have partition information

#get the border in an array of bPoint's
myBorder = myBorder(partVert,numPart,Nx,rank)

#get a dictionary of the neighboring parts and an inverted version
#where the keys are global rank and local rank respectively
myNeighbors,inv_n = findN(myBorder)

myPart,intOffset = gp.getPartitionAndOffset(partVert,rank);

#find the inner points
myInner = findInner(myPart,myBorder)

#make an even and odd data set so updating the board is possible
myDataE = makeData(myPart)
myDataO = deepcopy(myDataE)
"""
buffer = [0 for i in range(len(myDataE))]
for i in range(len(myDataE)):
  buffer[i] = myDataE[i].data
"""

#makes the initial sending array(use updateSend() for following calls)
send = sendIt(myDataE,myBorder,myNeighbors,inv_n)
get = deepcopy(send)

#START OF SIMULATION!!!!!!!
##################################################################
gen = 100
for i in range(gen):

  #sends all of the information out to the different ranks
  for j in range(len(send)):
    req = comm.isend(send[j],dest=(myNeighbors[j]), tag=(rank+2019))

  #while info is sent do the inner parts
  if i%2==0:
    calcInner(myPart,myInner,myDataE,myDataO,Nx)
  else:
    calcInner(myPart,myInner,myDataO,myDataE,Nx)


  #receives the info and stores it
  for j in range(len(get)):
    req = comm.irecv(source=(myNeighbors[j]), tag=(myNeighbors[j]+2019))
    get[j] = req.wait()

  #Use the collected info to calculate border cases
  if i%2==0:
    calcBorder(myPart,myBorder,get,myDataE,myDataO,Nx)
    """
    writeBuffer(buffer,myDataE)
    fname = "data_" + str(i) + ".b_dat"
    amode = MPI.MODE_WRONLY | MPI.MODE_CREATE #<- bit mask for file mode
    fh = MPI.File.Open(comm,fname,amode)
    offset = intOffset*np.dtype(np.int32).itemsize
    fh.Write_at_all(offset,np.array(buffer).astype('int32'))
    fh.Close()
    """
    updateSend(myDataE, send)
  else:
    calcBorder(myPart,myBorder,get,myDataO,myDataE,Nx)
    """
    writeBuffer(buffer,myDataO)
    fname = "data_" + str(i) + ".b_dat"
    amode = MPI.MODE_WRONLY | MPI.MODE_CREATE #<- bit mask for file mode
    fh = MPI.File.Open(comm,fname,amode)
    offset = intOffset*np.dtype(np.int32).itemsize
    fh.Write_at_all(offset,np.array(buffer).astype('int32'))
    fh.Close()
    """
    updateSend(myDataO, send)

  

##################################################################


#uncomment when ready for performance testing
print (time.time() - start_time)

"""
# write to disk a partition map giving the global point number in order for all partitions
filename = 'partMap.b_dat'
amode = MPI.MODE_WRONLY | MPI.MODE_CREATE #<- bit mask for file mode
fh = MPI.File.Open(comm,filename,amode)
offset = intOffset*np.dtype(np.int32).itemsize
fh.Write_at_all(offset,myPart)
fh.Close()

comm.Barrier() # make sure everyone is done
# rank 0 load this file and print to screen
if rank==0:
  partMap = np.fromfile(filename,dtype='int32')
  for i in range(gen):
    vtkname = "state_" + str(i) + ".vtk"
    dataName = "data_" + str(i) + ".b_dat"
    data_i = np.fromfile(dataName,dtype='int32')
    data = np.zeros_like(data_i)
    data[partMap] = data_i
    dims = [Nx, Nx,1]
    origin = [0,0,0]
    spacing = [1,1,1]
    writeVTK(data,'state',vtkname,dims,origin,spacing)
"""
