import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pickle
import math
from math import e
import pylab as pl
import matplotlib.ticker as mticker
N = 40
F = 1



class LORENZ96:
    def __init__(self, N, F):
        self.N = N
        self.F = F
        self.params = [self.N, self.F]

    def lorenz96(self,x):
        d = np.zeros(N, dtype = np.float64)
        d[0] = (x[1] - x[N-2]) * x[N-1] - x[0]
        d[1] = (x[2] - x[N-1]) * x[0] - x[1]
        d[N-1] = (x[0] - x[N-3]) * x[N-2] - x[N-1]
        for i in range(2, N-1):
            d[i] = (x[i+1] - x[i-2]) * x[i-1] - x[i]
        # Add the forcing term
        d = d + F

        # Return the state derivatives
        return d
    
    #def run(self, x0, t, t_output):
    #    x = solve_ivp(self.lorenz96, (t[0],t[-1]), x0, t_eval= t_output) 
    #    return x
    
    def run_rk4(self, time_start, time_end, dt, state_all0):
        n = np.arange(time_start, time_end+1e-5, dt)
        y = state_all0
        y = np.array(y)
        stacked_array = y
        for i in range(len(n)-1):
            dx1 = np.array(self.lorenz96(y))
            x1 = y+ dx1 * dt * 0.5
            dx2 = np.array(self.lorenz96(x1))
            x2 = y + dx2 * dt * 0.5
            dx3 = np.array(self.lorenz96(x2))
            x3 = y + dx3 * dt
            dx4 = np.array(self.lorenz96(x3))
            y = y + (dt / 6.0) *(dx1 + 2.0 * dx2 + 2.0 * dx3 + dx4)
            time_start = time_start + dt
            stacked_array = np.vstack((stacked_array, y))
        return stacked_array
            
        
   
        

 





