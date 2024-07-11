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

try :
    from full_epsilon import *
except :
    from DAQ_Python.full_epsilon import *




### input
location='D:/Users/Manips/Documents/DATA/FRICS/2022/2022-06-24/'
indice = 5









### Load data

pattern='osc_{}' + '_{}_'.format(indice)


file=pattern.format(1)+'epsilon_time_xx_yy_xy.npy'
file2=pattern.format(2)+'epsilon_time_xx_yy_xy.npy'
print(file)

reversed=True
Full_reversed = -1

n_avg=10
n_rolling = 100
results = np.load(location+'npy/'+file)
results=smooth(results,n_rolling)
results2 = np.load(location+'npy/'+file2)
results2=smooth(results2,n_rolling)

### Clean Data

def data_preparation(results,n_avg=1,reversed=False):
    time=results[0]
    fs = round(1/np.median(time[1:]-time[:-1]))
    data_tot = results[1:]
    data_tot_avg=[]
    nevent=results.shape[-1]//2
    event_width=int(6e-3*fs)
    for i in range(len(data_tot)):
        dat=data_tot[i]
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
fs = round(1/np.median(time[1:]-time[:-1]))

center = len(time)//2
width=int(0.01*fs)
time=time[center-width:center+width]
data_tot=Full_reversed*data_tot[:,center-width:center+width]

time2,data_tot2 = data_preparation(results2,n_avg=n_avg,reversed=reversed)
fs2 = round(1/np.median(time2[1:]-time2[:-1]))

center2 = len(time2)//2
width2=int(0.01*fs2)
time2=time2[center2-width2:center2+width2]
data_tot2=Full_reversed*data_tot2[:,center2-width2:center2+width2]



### Plot the full gages

fig, axes = plt.subplots(2, 3,sharex=True)
axes=axes.T
axes[0,0].plot(time,data_tot[0],label=r'$\varepsilon_{xx}$')
axes[1,0].plot(time,data_tot[1],label=r'$\varepsilon_{yy}$')
axes[2,0].plot(time,data_tot[2],label=r'$\varepsilon_{xy}$')

axes[2,0].set_xlabel("time (s)")
axes[1,0].set_ylabel('Strain')


axes[0,1].plot(time2,data_tot2[0],label=r'$\varepsilon_{xx}$')
axes[1,1].plot(time2,data_tot2[1],label=r'$\varepsilon_{yy}$')
axes[2,1].plot(time2,data_tot2[2],label=r'$\varepsilon_{xy}$')

axes[2,1].set_xlabel("time (s)")
axes[1,1].set_ylabel('Strain')


for ax in axes.flatten():
    ax.grid(which='both')
    ax.legend()

plt.tight_layout()
plt.savefig(location+'fig/'+'figure_fullgage_event_{}.png'.format(i))
plt.show()

### Plot 4 gages along the interface

labels = [2,5,8,14]
file=pattern.format(1)+'time_u21_u22.npy'
file2=pattern.format(2)+'time_u21_u22.npy'
print(file)

reversed=True
Full_reversed = -1

n_avg=1
n_rolling = 100
results = 1000*np.load(location+'npy_raw_U2/'+file)
results=results[:,len(results[0])//2-len(results[0])//10:len(results[0])//2+len(results[0])//10]
results=smooth(results,n_rolling)
results2 = 1000*np.load(location+'npy_raw_U2/'+file2)
results2=results2[:,len(results2[0])//2-len(results2[0])//10:len(results2[0])//2+len(results2[0])//10]
results2=smooth(results2,n_rolling)



fig, axes = plt.subplots(4, 1,sharex=True)

axes[0].plot(results[0],results[1]-np.average(results[1][0:100]),label='chan {}'.format(labels[0]))
axes[1].plot(results[0],results[2]-np.average(results[2][0:100]),label='chan {}'.format(labels[1]))
axes[2].plot(results2[0],results2[1]-np.average(results2[1][0:100]),label='chan {}'.format(labels[2]))
axes[3].plot(results2[0],results2[2]-np.average(results2[2][0:100]),label='chan {}'.format(labels[3]))

for ax in axes.flatten():
    ax.grid(which='both')
    ax.legend()

ax.set_xlabel("time (ms)")
ax.set_ylabel("mV")

plt.tight_layout()
plt.savefig(location+'fig/'+'figure_propagation_event_{}.png'.format(i))

plt.show()



