### Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from matplotlib import rc
from scipy.optimize import curve_fit

import os

try :
    from Python_DAQ import *
except :
    from DAQ_Python.Python_DAQ import *



### Constants
C_r=1255
C_s=1345
C_d=2332
nu=0.335
E=5.651e9


### Load data
location='D:/Users/Manips/Documents/DATA/FRICS/2022/2022-02-11/200kg/npy/'


files = sorted(os.listdir(location))
file=files[4]
print(file)

reversed=True

n_avg=1
n_rolling = 10
results = np.load(location+file)
results=smooth(results,n_rolling)

### Clean Data

def event_finder(signal,fs):
    n_smooth=int(0.00001*fs)
    a=np.diff(avg_bin(signal,n_smooth))
    #plt.plot(a)
    #plt.show()
    return(n_smooth*np.argmax(np.abs(a)))

def data_preparation(results,n_avg=1,reversed=False):
    time=results[0]
    fs = round(1/np.median(time[1:]-time[:-1]))
    data_tot = results[1:]
    data_tot_avg=[]
    nevent=event_finder(results[1],fs)
    event_width=int(6e-5*fs)
    for i in range(len(data_tot)):
        dat=data_tot[i]
        if i ==2:
            dat= dat-np.average(dat[nevent+event_width-10:nevent+event_width+10])
        if i!=2:
            dat=dat-np.average(dat[:nevent-2*event_width])
        data_tot_avg.append(avg_bin(dat,n_avg))
    data_tot_avg=np.array(data_tot_avg)
    if reversed:
        data_tot_avg[2]=-data_tot_avg[2]
    time = time[::n_avg]
    while len(time)>data_tot_avg.shape[-1]:
        time=time[:-1]
    return(time,data_tot_avg)




time,data_tot = data_preparation(results,n_avg=n_avg,reversed=reversed)
fs = round(1/np.median(time[1:]-time[:-1]))

center = event_finder(data_tot[0],fs)
width=int(0.001*fs)
time=time[center-width:center+width]
data_tot=data_tot[:,center-width:center+width]

### Simply plot data

fig, axes = plt.subplots(3, 1,sharex=True)

axes[0].plot(time,data_tot[0],label=r'$\varepsilon_{xx}$')
axes[1].plot(time,data_tot[1],label=r'$\varepsilon_{yy}$')
axes[2].plot(time,data_tot[2],label=r'$\varepsilon_{xy}$')

axes[2].set_xlabel("time (s)")
axes[1].set_ylabel('Strain')


for ax in axes:
    ax.grid(which='both')
    ax.legend()

plt.tight_layout()
plt.show()



### Fitting function

with open("./DAQ_Python/full_epsilon.py") as f:
    exec(f.read())
    f.close()


def toopt(time,v,y_pos,t_0,Gamma):
    """
    Dummy function serving as an intermediary for scipy.curve_fit
    """
    args=v,y_pos,t_0,Gamma
    return(full_epsilon(time,args).reshape(-1))

### Optimization parameters

nevent=event_finder(data_tot[0],fs)
event_width=int(6e-5*fs)

#set bounds
bounds=[
[ 10     ,1100  ], #v
[ 0.002 ,0.004 ], #y_pos
[time[nevent]-0.0005  ,time[nevent]+0.0005 ], #t_0
[ 0.1    ,2     ]] #Gamma
#[-0.00000002 ,0.00000002]] #\tau_r shift
bounds=np.array(bounds)
bounds=bounds.transpose()

# set proportion of the data to take into account


start=nevent-5*event_width
end=nevent+3*event_width

names=['v     ','y_pos ','t_0   ','Gamma ']



### Optimization computation

args=curve_fit(toopt, time[start:end], data_tot[:,start:end].reshape(-1),bounds=bounds)
args=np.array((args[0],args[1][0],args[1][1]))

for i in range(len(args[0])):
    print(names[i],args[0,i])

print('\n\n')
### Compute results

xx,yy,xy = full_epsilon(time,args[0])
v,y_pos,t_0,Gamma=args[0]
x=-v*(time-t_0)

### Plot
fig, axes = plt.subplots(3, 1,sharex=True)

axes[0].plot(time,data_tot[0],label=r'$\varepsilon_{xx}$')
axes[1].plot(time,data_tot[1],label=r'$\varepsilon_{yy}$')
axes[2].plot(time,data_tot[2],label=r'$\varepsilon_{xy}$')

axes[0].plot(time,xx,label=r'$\varepsilon_{xx}$ fit')
axes[1].plot(time,yy,label=r'$\varepsilon_{yy}$ fit')
axes[2].plot(time,xy,label=r'$\varepsilon_{xy}$ fit')

axes[2].set_xlabel("time (s)")
axes[1].set_ylabel('Strain')

for i in range(3):
    axes[i].axvline(time[end],linestyle="--",color='k')
    axes[i].axvline(time[start],linestyle="--",color='k')


for ax in axes:
    ax.grid(which='both')
    ax.legend()

plt.tight_layout()
plt.show()























