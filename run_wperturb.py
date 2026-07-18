import numpy as np
import os
import sys
from active_fluid import active_fluid

omega = float(sys.argv[1])
domega = float(sys.argv[2])
seed = int(sys.argv[3])
state = str(sys.argv[4])
# direc_save= 'data/perturb/1/'
# os.makedirs(direc_save,exist_ok=True)
np.random.seed(seed)


AF1 = active_fluid(N_ptcl=200000,Fs=1000)
AF1.u = 30
AF1.Dr = 1
AF1.alpha = 1
AF1.lamb = 1*AF1.u
AF1.l_passive = 10
AF1.L = 40
AF1.R = 5
AF1.Rb = 1
AF1.mu_T = 0.1
AF1.mu_R = np.array([30])
# AF1.RA = np.array([AF1.R])/2
AF1.set_zero()

AF1.mode='forced'
AF1.omega=omega+domega

for i in range(3000):
    AF1.time_evolve()

len_traj = 15000
tau_traj = np.zeros(len_traj)
save_dict={}
# save_dict['S1'] = S1

AF1.omega=omega

for i in range(len_traj):
    tau_traj[i] = AF1.time_evolve()[0]
save_dict['tau_traj'] = tau_traj
save_dict['domega'] = domega
save_dict['omega'] = omega

np.savez(state, **save_dict)

