import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from active_fluid import active_fluid

I_R = float(sys.argv[1])
seed = int(sys.argv[2])
state = str(sys.argv[3])
# direc_save= 'data/rotating/1/'
# os.makedirs(direc_save,exist_ok=True)
np.random.seed(seed)

# AF1 = active_fluid(N_ptcl=200000,Fs=1000)
# AF1 = active_fluid(N_ptcl=50000,Fs=1000)
# AF1 = active_fluid(N_ptcl=20000,Fs=1000)
# AF1 = active_fluid(N_ptcl=10000,Fs=1000)
AF1 = active_fluid(N_ptcl=5000,Fs=1000)

AF1.u = 30
AF1.Dr = 1
# AF1.alpha = 1
AF1.lamb = 1*AF1.u
AF1.l_passive = 10
AF1.L = 40
AF1.R = 5
AF1.Rb = 1
AF1.mu_T = 0.1
AF1.I_R = np.array([1])*I_R
# AF1.RA = np.array([AF1.R])/2
AF1.set_zero()


# AF1 = active_fluid(N_ptcl=100000,Fs=1000)
# # AF1.u = 50
# AF1.u = 20
# AF1.Dr = 1
# AF1.alpha = 1
# AF1.lamb = 1*AF1.u
# AF1.l_passive = 10
# AF1.L = 40
# AF1.R = 3
# AF1.Rb = 1
# AF1.mu_T = 0.1
# AF1.mu_R = np.array([30])
# # AF1.RA = np.array([AF1.R])/2
# AF1.set_zero()

AF1.mode='free'
AF1.omega=0

for i in range(3000):
    AF1.time_evolve()

len_traj = 15000
tau_traj = np.zeros(len_traj)
omega_traj = np.zeros(len_traj)
Theta_traj = np.zeros(len_traj)
save_dict={}
# save_dict['S1'] = S1
save_dict['tau_traj'] = tau_traj
save_dict['omega_traj'] = omega_traj
save_dict['Theta_traj'] = Theta_traj
save_dict['N_ptcl'] = AF1.N_ptcl
save_dict['I_R'] = I_R
save_dict['Fs'] = AF1.Fs
save_dict['dt'] = AF1.dt
save_dict['u'] = AF1.u
save_dict['Dr'] = AF1.Dr
save_dict['L'] = AF1.L
save_dict['R'] = AF1.R
save_dict['Rb'] = AF1.Rb

for i in range(len_traj):
    for _ in range(50):
        tau_traj[i] += np.sum(AF1.time_evolve())/50
    omega_traj[i] = AF1.omega[0]
    Theta_traj[i] = AF1.Theta[0]


np.savez(state, **save_dict)
