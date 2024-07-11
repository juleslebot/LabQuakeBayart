### Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from matplotlib import rc
from scipy.optimize import curve_fit
from matplotlib.widgets import Slider, Button
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
location='D:/Users/Manips/Documents/DATA/FRICS/2022/2022-06-07/npy/'


files = sorted(os.listdir(location))
#file=files[3]
file='ens_2022_06_07_081311_2_R10_epsilon_time_xx_yy_xy.npy'
print(file)

reversed=False
Full_reversed = 1

n_avg=1
n_rolling = 1
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
    nevent=results.shape[-1]//2 #event_finder(results[1],fs)
    event_width=int(6e-3*fs)
    for i in range(len(data_tot)):
        dat=data_tot[i]
        #if i ==2:
        #    dat= dat-np.average(dat[nevent+event_width-10:nevent+event_width+10])
        #if i!=2:
        if True:
            dat=dat-np.average(dat[nevent-3*event_width:nevent-1*event_width])
        data_tot_avg.append(avg_bin(dat,n_avg))
    data_tot_avg=np.array(data_tot_avg)
    if reversed:
        data_tot_avg[2]=-data_tot_avg[2]
    time = time[::n_avg]
    while len(time)>data_tot_avg.shape[-1]:
        time=time[:-1]
    return(time,data_tot_avg)




time,data_tot = data_preparation(results,n_avg=n_avg,reversed=reversed)
time=time-time[0]
fs = round(1/np.median(time[1:]-time[:-1]))

center = len(time)//2 #event_finder(data_tot[0],fs)
width=int(0.05*fs)
time=time[center-width:center+width]
data_tot=Full_reversed*data_tot[:,center-width:center+width]

### Simply plot data
print(np.std(data_tot[0][:1000]))

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
#plt.show()

























### Load data
location='D:/Users/Manips/Documents/DATA/FRICS/2022/2022-05-25/xp1/npy/'


files = sorted(os.listdir(location))
#file=files[3]
file='osc_1_1_epsilon_time_xx_yy_xy.npy'
print(file)

reversed=True
Full_reversed = -1

n_avg=5
n_rolling = 1
results = np.load(location+file)
results=smooth(results,n_rolling)


time,data_tot = data_preparation(results,n_avg=n_avg,reversed=reversed)
fs = round(1/np.median(time[1:]-time[:-1]))
time=time-time[0]

center = len(time)//2 #event_finder(data_tot[0],fs)
width=int(0.01*fs)
time=time[center-width:center+width]
data_tot=Full_reversed*data_tot[:,center-width:center+width]


print(np.std(data_tot[0][:1000]))





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
