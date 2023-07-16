#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
from class_state_vec import state_vector
from class_obs import obs_da
import pickle
import numpy.matlib
from scipy import linalg

class da_system:
    
    def __init__(self, x0 = [], yo = [], t0 = 0, dt = 0, state_vector =[], obs_data = [], acyc_step = 1):
        self.xdim = np.size(x0)
        self.ydim = np.size(yo)
        self.edim = 0
        self.x0 = x0
        self.t0 = t0
        self.dt = dt
        self.X0 = x0  # capital X??
        self.t = t0
        self.acyc_step = acyc_step
        self.dtau = dt * acyc_step
        self.fcst_step = acyc_step
        self.fcst_dt = dt
        self.maxit = 0
        self.statedim = 0
        self.paradim = 0
        self.B = np.matrix(np.identity(self.xdim))
        self.R = np.matrix(np.identity(self.ydim))
        self.H = np.matrix(np.identity(self.xdim))  #same as B?
        #self.Ht = (self.H).transpose()
        self.SqrtB = []
        self.state_vector = state_vector
        self.obs_data = obs_data
        self.method = ''
        self.KH = []
        self.khidx = []
        self.das_bias_init = 0
        self.das_sigma_init = 0.1
        
    def __str__(self):
        print('xdim= ', self.xdim)
        print('ydim= ', self.ydim)
        print('x0= ', self.x0)
        print('t0= ', self.t0)
        print('dt= ', self.dt)
        print('t=', self.t)
        print('acyc_step=', self.acyc_step)
        print('dtau= ', self.dtau)
        print('fcst_step= ', self.fcst_step)
        print('fcst_dt=', self.fcst_dt)
        print('B = ')
        print(self.B)
        print('H = ')
        print(self.H)
        print('state_vector = ')
        print(self.state_vector)
        print('obs_data = ')
        print(self.obs_data)
        print('method = ')
        print(self.method)
        return 'type::da_system'
    
    def setMethod(self, method):
        self.method = method
        
    def getMethod(self):
        return self.method
    
    def setStateVector(self, sv):
        self.state_vector = sv
        
    def getStateVector(self):
        return self.state_vector
        
    def setObsData(self, obs):
        self.obs_data = obs
        
    def getObsData(self):
        return self.obs_data
    
    def updata(self, B =[0], R=[0], H =[0], t = [0], x0 = [0]):
        self.B = B
        self.R = R
        self.H = H
        self.t = t
        #self.Ht = H.transpose()
        self.x0 = x0
        
    def setC(self,C):
        self.C = np.matrix(C)
    
    def getC(self):
        return self.C
    
    def setB(self,B):
        self.B = np.matrix(B)
        nr, nc = np.shape(B)
        self.xdim = nr
        self.SqrtB = linalg.sqrtm(self.B)
        
    def setSqrtB(self, X):
        self.SqrtB = np.matrix(X)
        self.B = self.SqrtB * self.SqrtB.T   # xxxx.T   transpose xxxx
        nr, nc = np.shape(X)
        self.xdim = nr
        self.edim = nc
        
    def getB(self):
        return self.B
        
    def getR(self):
        return self.R
    
    def setR(self, R):
        #self.R = np.matrix(R)
        self.R = R
        #self.Rinv = np.linalg.inv(R)
        
    def getH(self):
        return self.H
    
    def setH(self, H):
        self.H = H
        #self.Ht = np.transpose(self.H)
        
    def setKH(self, KH, khidx):
        self.KH = KH
        self.khidx = khidx
        
    def getKH(self, KH, khidx):
        return self.KH, self.khidx
        
        
    #def reduceYdim
    def reduceYdim(self,yp):
        #   print('reduceYdim:')
        #   print('yp = ', yp)
        self.ydim = len(yp)
        self.setH(self.H[yp,:])
        
        R = self.R
        
        #R = R[yp,:]
        #R = R[:,yp]
        self.setR(R)
   
    
    def compute_analysis(self, Xb, yo, Yb, ym , xm,R):
        method = self.method
        if method == 'skip':
            xa = xb
            KH = np.identity(self.xdim)
        elif method == 'EnKF':
            xa= self.ETKF(Xb, yo, Yb, ym , xm,R)
                
        else:
            print('Unrecognized DA method.')
            raise SystemExit
        return xa
        
    def initEns(self, x0, mu = 0, sigma = 0.1, edim = 4,separate = 'undecided'): # edim
        x0 = np.matrix(x0).flatten().T
        mu = np.matrix(mu).flatten().T
        if separate == 'no':
            xdim = len(x0)
            Xrand = np.random.normal(mu, sigma, (xdim, edim))
            Xrand = np.matrix(Xrand)
            #print('Xrand_all = ')
            #print(Xrand)
            rand_mean = np.mean(Xrand, axis = 1) - mu
            #print('rand_mean = ')
            #print(rand_mean)
            rmat = np.matlib.repmat(rand_mean, 1, edim)
            #Xrand = Xrand - rmat
            rmat = np.matlib.repmat(x0, 1, edim )
            X0 = np.matrix(Xrand + rmat)
        elif separate == 'state':
            #Purterbation only applied to state variables
            Xrand = np.random.normal(mu,10*sigma,(self.statedim,edim))  # #states * #ensemble
            Xrand = np.matrix(Xrand)
            print('Xrand_state = ')
            print(Xrand)
            rand_mean = np.mean(Xrand, axis = 1) - mu
            print('rand_mean = ')
            print(rand_mean)
            rmat = np.matlib.repmat(rand_mean, 1, edim)
            Xrand = Xrand - rmat
            
            """
            Xrand_complete = np.zeros((self.paradim,edim))
            Xrand_complete = np.matrix(Xrand_complete)
            Xrand = np.concatenate((Xrand, Xrand_complete))
            rmat = np.matlib.repmat(x0,1,edim)
            """
            rmat = np.matlib.repmat(x0[0:4],1,edim)
            X0 = np.matrix(Xrand + rmat)
            #print('Perturbated X0 = ', X0)
        elif separate == 'parameter':
            #Purterbation only applied to parameter variables
            Xrand_beta = np.random.normal(mu,0.0000001*sigma,(2,edim))  # #parameters * #ensemble
            Xrand_beta = np.matrix(Xrand_beta)
            Xrand_others = np.random.normal(mu,sigma,(4,edim))  # #parameters * #ensemble
            Xrand_others = np.matrix(Xrand_others)
            Xrand_parameter = np.concatenate((Xrand_beta,Xrand_others))
            
            rand_mean_para = np.mean(Xrand_parameter, axis = 1) - mu
            print('rand_mean_para = ')
            print(rand_mean_para)
            
            rmat = np.matlib.repmat(rand_mean_para, 1, edim)
            Xrand_parameter = Xrand_parameter - rmat
            
            print('Xrand_parameter = ')
            print(Xrand_parameter)
            """
            Xrand_complete = np.zeros((self.statedim,edim))
            Xrand_complete = np.matrix(Xrand_complete)
            Xrand = np.concatenate((Xrand_complete,Xrand))
            rmat = np.matlib.repmat(x0,1,edim)
            """
            rmat = np.matlib.repmat(x0[4:10],1,edim)
            X0 = np.matrix(Xrand_parameter + rmat)  # #parameters
            #print('Perturbated X0 = ', X0) 
        else:
            print('Unrecognized separate command.')
            raise SystemExit
            
        return X0
    
    
        
#---------------------------------------------------------------------------------------------------
    def ETKF(self, Xb, yo, Yb, ym , xm, R):
        
        
#---------------------------------------------------------------------------------------------------
# Use ensemble of states to estimate background error covariance
        verbose=False # don't show detail information.
        EPES = False

    # Make sure inputs are matrix and column vector types
        Xb = np.matrix(Xb)   
        yo = np.matrix(yo).flatten().T  # #obsversed states*1
        #print('Xb')
        #print(Xb) 
    # Get system dimensions
        nr,nc = np.shape(Xb)
        xdim = nr  # == self.xdim  4
        edim = nc   #4
        ydim = len(yo)   #row  #obsversed states
        #Hl = self.H[0:39,0:39]
      

    # Apply observation operator to forecast ensemble
        '''
        Yb = np.matrix(np.zeros([ydim,edim]))  # #obsversed states *4
 
        for i in range(edim):
            Hl = self.H
            Yb[:,i] = np.dot(Hl,Xb[:,i])
        '''
                  
            
    # Convert ensemble members to perturbations
        '''
        xm = np.mean(Xb,axis=1)
        ym = np.mean(Yb,axis=1)
       
        Xb = Xb - np.matlib.repmat(xm, 1, edim) # 4*4 -4*4
        Yb = Yb - np.matlib.repmat(ym, 1, edim) ##obsversed states *1
        '''
        
    # Compute R^{-1}
        #R = self.R
        Rinv = np.linalg.inv(R)
        
    # Compute the weights

    #----
    # stage(4) Now do computations for lxk Yb matrix
    # Compute Yb^T*R^(-1)
    #----
        Ybt = np.transpose(Yb) # 1* obsversed states
       
        C = np.dot(Ybt,Rinv) #Yb^T*R^(-1)
     

        if verbose:
           print ('C = ')
           print (C) 
       

    #----
    # stage(5) Compute eigenvalue decomposition for Pa
    # Pa = [(k-1)I/rho + C*Yb]^(-1)
    #----
        I = np.identity(edim) #4*4
        rho = 1.08  # multiplicative inflation
        eigArg = (edim-1)*I/rho + np.dot(C,Yb)
        #print('eigArg = ', eigArg)
 
        lamda,P = np.linalg.eigh(eigArg)
        #print('lambda = ', lamda)

        if verbose:
           print ('lamda = ')
           print (lamda)
           print ('P = ')
           print (P)

        Linv = np.diag(1.0/lamda)

        if verbose:
           print ('Linv = ')
           print (Linv)

        PLinv = np.dot(P,Linv)

        if verbose:
           print ('PLinv = ')
           print (PLinv)

        Pt = np.transpose(P)

        if verbose:
           print ('Pt = ')
           print (Pt)

        Pa = np.dot(PLinv, Pt)
        #print('Pa = ',Pa)

        if verbose:
           print ('Pa = ')
           print (Pa) 

    #----
    # stage(6) Compute matrix square root
    # Wa = [(k-1)Pa]1/2
    #----
        Linvsqrt = np.diag(1/np.sqrt(lamda))

        if verbose:
           print ('Linvsqrt = ')
           print (Linvsqrt)

        PLinvsqrt = np.dot(P,Linvsqrt)

        if verbose:
           print ('PLinvsqrt = ')
           print (PLinvsqrt)

        Wa = np.sqrt((edim-1)) * np.dot(PLinvsqrt,Pt)  #  edim*edim
        #print ('Wa = ')
        #print (Wa)
        trace_pa = np.trace(Pa)
        #print('trace_pa=', trace_pa)
        lamde = np.sqrt(edim/((edim-1)*trace_pa))
        Xa_d = np.dot(Xb,Wa)
        
        if EPES:
           Wa = lamde * Wa
           Xa_d_p = np.dot(Xb[-1,:],Wa)
           Xa_para_spread = np.dot(Xa_d_p,Xa_d_p.T)
           print('Xa_para_spread = ', Xa_para_spread)
 
           Xa_d = np.concatenate((Xa_d[0:3,:],Xa_d_p))
           Xa_d_spread = (1/(edim-1)) * np.trace(np.dot(Xa_d,Xa_d.T))
           print('Xa_d_spread = ', Xa_d_spread)
           """
           Wa = Wa - np.matlib.repmat(np.mean(Wa, axis = 0),edim,1)
           Wa_var = np.var(Wa, axis = 0)
           Wa_var_sum = np.sum(Wa_var)
           infl = lamde * np.sqrt(1 / Wa_var_sum)
           Wa = Wa * infl
           """
           #print('infWa = ', Wa)    

        if verbose:
           print ('Wa = ')
           print (Wa)

    #----
    # stage(7) Transform back
    # Compute the mean update
    # Compute wabar = Pa*C*(yo-ybbar) and add it to each column of Wa
    #----
        d = yo-ym
        Cd = np.dot(C,d)

        if verbose:
           print ('Cd = ')
           print (Cd)

        wm = np.dot(Pa,Cd) #k x 1
        Xa = Xa_d + np.dot(Xb,np.matlib.repmat(wm, 1, edim)) + np.matlib.repmat(xm, 1, edim)
        
        if verbose:
           print ('wm = ')
           print (wm)

    # Add the same mean vector wm to each column
#   Wa = Wa + wm[:,np.newaxis] #STEVE: make use of python broadcasting to add same vector to each column
        #Wa = Wa + np.matlib.repmat(wm, 1, edim)

        if verbose:
           print ('Wa = ')
           print (Wa)
        
        
    #----
    # stage(8)
    # Compute the perturbation update
    # Multiply Xb (perturbations) by each wa(i) and add xbbar
    #----

    # Add the same mean vector wm to each column
#   Xa = np.dot(Xb,Wa) + xm[:,np.newaxis]
        
        #Xa = np.dot(Xb,Wa) + np.matlib.repmat(xm, 1, edim)
        
        if verbose:
           print ('Xa = ')
           print (Xa)

    # Compute KH:
        RinvYb = np.dot(Rinv,Yb)
        IpYbtRinvYb = ((edim-1)/rho)*I + np.dot(Ybt,RinvYb)
        IpYbtRinvYb_inv = np.linalg.inv(IpYbtRinvYb)
        YbtRinv = np.dot(Ybt,Rinv)
        K = np.dot( Xb, np.dot(IpYbtRinvYb_inv,YbtRinv) )
        #KH = np.dot(K,Hl)
        return Xa
    def save(self,outfile):
        with open(outfile,'wb') as output:
            pickle.dump(self,output,pickle.HIGHEST_PROTOCOL)
          
    def load(self,infile):
        with open(infile,'rb') as input:
            das = pickle.load(input)
            return das
        
        
    
    


