#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#state_vector
import numpy as np
import pickle

class state_vector:
    def __init__(self, al = [0], t = [0], name = 'uninitialized', paraname = 'unknown'):
        self.tdim = np.size(t)
        self.al = al
        self.aldim = np.size(al)
        self.statedim = np.size(al[0:3])
        self.pdim = np.size(al[3:3])
        self.x0 = al[0:3]
        self.p0 = al[3:3]
        self.t = t
        self.name = name
        self.trajectory = np.zeros([self.tdim, self.aldim])
        self.Enstrajectory = np.zeros([self.tdim, self.aldim, 7])
        self.Xatrajectory = np.zeros([self.tdim,self.aldim,7])
        self.paraname = paraname
        
    def __str__(self):
        print(self.name)
        print('Number of states and parameters')
        print(self.aldim)
        print('Parameters:')
        print(self.p0)
        print('Number of parameters:')
        print(self.pdim)
        print('Initial condition:')
        print(self.x0)
        print('Number of states:')
        print(self.statedim)
        print('Trajectory:')
        print(self.trajectory)
        print('EnsTrajectory:')
        print(self.Enstrajectory)
        return self.name
    
    def setName(self,name):
        self.name = name
    
    def getTrajectory(self):
        return self.trajectory
        
    def setTrajectory(self,states):
        self.trajectory = states
        
    def getEnsTrajectory(self):
        return self.Enstrajectory
    
    def setEnsTrajectory(self,enstates):
        self.Enstrajectory = enstates
        
    def setXaTrajectory(self,Xa_history):
        self.Xatrajectory = Xa_history
       
    def getXaTrajectory(self):
        return self.Xatrajectory
        
    def setParaName(self,paraname):
        self.paraname = paraname
        
    def getParaName(self):
        return self.paraname    
        
    def getTimes(self):
        return self.t
    
    def setStateDim(self,statedim):
        self.statedim = statedim
        
    def setParaDim(self,paradim):
        self.pdim = paradim
    
    def getStateDim(self):
        return self.statedim
    
    def getParaDim(self):
        return self.pdim
    
    def save(self, outfile):
        with open(outfile, 'wb') as output:
            pickle.dump(self, output)
        
    def load(self, infile):
        with open(infile, 'rb') as input:
            sv = pickle.load(input)
        return sv



