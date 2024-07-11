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

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif"})



### Constants
C_r=1255
C_s=1345
C_d=2332
nu=0.335
E=5.651e9


### Load data
location = "D:/Users/Manips/Documents/DATA/FRICS/Data_Jeru_ForYohann/data_2/"
fitloc = "D:/Users/Manips/Documents/DATA/FRICS/Data_Jeru_ForYohann/data_2/data_2_fit.npy"

files = sorted(os.listdir(location))
#file=files[3]
file='data_2_xp.npy'
print(file)

reversed=False
Full_reversed = 1

n_avg=1
n_rolling = 2
results = np.load(location+file)
results=smooth(results,n_rolling)

fit = np.load(fitloc)

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
    event_width=len(time) #int(6e-3*fs)
    for i in range(len(data_tot)):
        dat=data_tot[i]
        if i ==2:
            dat= dat-np.average(dat[-len(dat)//10:])
        if i!=2:
            dat=dat-np.average(dat[:len(dat)//10])
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

center = len(time)//2 #event_finder(data_tot[0],fs)
width=len(time)//2 #int(0.01*fs)
time=time[center-width:center+width]
data_tot=Full_reversed*data_tot[:,center-width:center+width]



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

nevent=data_tot.shape[-1]//2#event_finder(data_tot[0],fs)
event_width=int(6e-4*fs)

#set bounds
bounds=[
[ 220    ,224 ], #v
[ 0.0034 ,0.0036 ], #y_pos
[-0.00025,0.00025],#t_0
[ 0.1 , 10     ]] #Gamma
bounds=np.array(bounds)
bounds=bounds.transpose()

# set proportion of the data to take into account


start=1750
end=1970

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
fig, axes = plt.subplots(1, 3,sharex=True,sharey=False,figsize=(9.5,3))

axes[0].plot(time*1e3,-data_tot[0]*1e6,label=r'$-\varepsilon_{xx}$',alpha=0.5,color='k',linewidth=0.5)
axes[1].plot(time*1e3,data_tot[1]*1e6,label=r'$\varepsilon_{yy}$',alpha=0.5,color='k',linewidth=0.5)
axes[2].plot(time*1e3,data_tot[2]*1e6,label=r'$\varepsilon_{xy}$',alpha=0.5,color='k',linewidth=0.5)


line1,=axes[0].plot(time*1e3,-xx*1e6,label=r'LEFM fit',linewidth=3,alpha=0.5,color='r')
line2,=axes[1].plot(time*1e3,yy*1e6,label=r'LEFM fit',linewidth=3,alpha=0.5,color='g')
line3,=axes[2].plot(time*1e3,xy*1e6,label=r'LEFM fit',linewidth=3,alpha=0.5,color='b')

axes[1].set_xlabel("time (ms)")
axes[0].set_ylabel('Strain (µm/m)')



for ax in axes:
    ax.grid(which='both')
    ax.legend()

plt.xlim(-0.15,0.15)
axes[0].set_ylim(-30,160)
axes[0].set_yticks([-25,0,25,50,75,100,125,150])

axes[1].set_ylim(-30,60)
axes[1].set_yticks([-25,0,25,50])

axes[2].set_ylim(-30,60)
axes[2].set_yticks([-25,0,25,50])

plt.xticks([-0.1,0,0.1])
plt.tight_layout()
#plt.show()



plt.savefig("test2.pdf",dpi=600)















