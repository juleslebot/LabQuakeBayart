from nptdms import TdmsFile
import matplotlib.pyplot as plt
import scipy
import numpy as np
import sys

filename = 'Sampling14-183mm.tdms'
path = 'E:/2023-2024/2024-05-15-acoustic-sensor-callibration/'
ch1,ch2,ch3 = open_data(filename,path,'cont')

g = 10 # m/s2
fs = 1e+6 # Hz
mass = 2.4e-4 # kg

ind = [2197000,2536000,2843000,3109000,3341000,3561000,3755000,3918000,4078000,4211000]
events0 = np.zeros((3,10,1200))
energy0 = np.zeros((3,10))
bounce_time = np.zeros((3,10))
for i in range(10):
    events0[0,i,:] = isolate_event(ch1[ind[i]-10000:ind[i]+10000],200,1000)
    events0[1,i,:] = isolate_event(ch2[ind[i]-10000:ind[i]+10000],200,1000)
    events0[2,i,:] = isolate_event(ch3[ind[i]-10000:ind[i]+10000],200,1000)
    bounce_time[0,i] = (ind[i]-10000+np.argmax(ch1[ind[i]-10000:ind[i]+10000]))/fs
    bounce_time[1,i] = (ind[i]-10000+np.argmax(ch2[ind[i]-10000:ind[i]+10000]))/fs
    bounce_time[2,i] = (ind[i]-10000+np.argmax(ch3[ind[i]-10000:ind[i]+10000]))/fs
    energy0[0,i] = compute_energy(events0[0,i,:])
    energy0[1,i] = compute_energy(events0[1,i,:])
    energy0[2,i] = compute_energy(events0[2,i,:])

heights = g/8*(bounce_time[:,:-1]-bounce_time[:,1:])**2
delta_E_mecanic = (heights[:,:-1]-heights[:,1:])*g*mass
delta_E_acoustic0 = energy0[:,:-1]-energy0[:,1:]



fig, ax1 = plt.subplots(3,1,sharex=True,sharey=True,constrained_layout = True)
for i in range(3):
    if i ==2:ax1[i].set_xlabel('Indice du rebond')
    ax1[i].set_ylabel('Energie capteur',color='b')
    ax1[i].plot(delta_E_acoustic0[i,:],'b',label="Delta E acoustic, Nbefore =200, Nafter=1500",marker="+",linestyle="-.",linewidth=.5)
    ax1[i].tick_params(axis='y', labelcolor='b')
    ax1[i].grid(which='both')
    ax2 = ax1[i].twinx()
    ax2.set_ylabel('Energie rebond [J]',color='orange')
    ax2.plot(delta_E_mecanic[i,:],'orange',label='E rebond ch'+str(i+1),marker="^",linestyle="-.",linewidth=.5)
    ax2.tick_params(axis='y', labelcolor='orange')
fig.tight_layout()
fig.set_size_inches(14,8)


alpha = np.zeros((10,3))

for i in range(3):
    alpha[:,i] = energy0[i,:]/energy0[i,0]
plt.figure()
plt.plot(alpha)
plt.show()