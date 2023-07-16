
# coding: utf-8

# In[1]:


from class_siir_mdl import SIIR
from class_state_vec import state_vector
import numpy as np

#------------------------------------------------------------------
# Setup initial state
#------------------------------------------------------------------

N = 13951636 #Total number of Tokyo population(Wiki)

name = 'x_nature'

t0=0
tf=90
#dt=0.001
dt=0.01
#state_all_0 = [N-1, 1, 0, 0, 0.0000001, 0.00000001, 1/20, 1/10, 0.03, 0.5]
#state_all_0 = [N-1, 1, 0, 0, 0.000000014, 0.000000012, 1/14, 1/7, 1/14, 0.5]
#state_all_0 = [N-1, 1, 0, 0, 1.4]
state_all_0 = [1, 0, 0, 1.4]  # after replace S as N - I1 - I2 - R
tvec = np.arange(t0, tf, dt)
TVEC = 100 * tvec  
#TVEC = 1000 * tvec  # float can not be used as indices
TVEC = TVEC.astype(int) # float can not be used as indices

#------------------------------------------------------------------
# Setup state vector object
#------------------------------------------------------------------
sv = state_vector(al=state_all_0, t=tvec, name=name)

#------------------------------------------------------------------
# Initialize the SIIR object
#------------------------------------------------------------------
smd = SIIR()

#------------------------------------------------------------------
# Run smd to generate a nature run with the specified parameters
#------------------------------------------------------------------
print('Run S-I1-I2-R model')
trajectory = smd.run(sv.al, sv.t)  # run odeint
#trajectory = smd.run_rk4(t0, tf, dt, state_all_0)   # run runge kutta 4
print(trajectory)
sv.setTrajectory(trajectory)
smd.plot_state1(tvec, trajectory[TVEC,:])
smd.plot_state2(tvec, trajectory[TVEC,:])
#smd.plot_state3(tvec, trajectory[TVEC,:])
#smd.plot_state4(tvec, trajectory[TVEC,:])
#smd.plot_state5(tvec, trajectory[TVEC,:])



#------------------------------------------------------------------
# Output the beginning and end states, and compute model climatology
#------------------------------------------------------------------
print(sv)

#------------------------------------------------------------------
# Store the nature run data
#------------------------------------------------------------------
outfile = name+'.pkl'
sv.save(outfile)

