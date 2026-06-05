import numpy as np
import os

omega_list = [0,0.3,0.6,1,1.5,2.0,2.5,3.0,3.5,4.0]#[1,2,3,4,5,6,7,8,9,10,11,12]#[0,0.1,0.05,0.02]#,0.01,0.005,0.002,0.001]
domega_list = [-0.1,-0.2,-0.3,0.1,0.2,0.3]
file = open('jobs.txt','a')
direc_save= 'data/wperturb/8/'
os.makedirs(direc_save,exist_ok=True)

for omega in omega_list:
    for domega in domega_list:
        for i in range(10):
            state = os.getcwd()+'/'+direc_save+str(omega)+'-'+str(domega)+'seed'+str(i)+'.npz'
            if os.path.exists(state):
                pass
            else:
                file.write('/pds/pds21/yunsik/miniconda3/bin/python run_wperturb.py %f %f %i %s  \n' % (omega,domega,i,state))
file.close()
