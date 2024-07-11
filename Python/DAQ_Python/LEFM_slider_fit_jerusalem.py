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
location = "D:/Users/Manips/Documents/DATA/FRICS/Data_Jeru_ForYohann/data_2/"
fitloc = "D:/Users/Manips/Documents/DATA/FRICS/Data_Jeru_ForYohann/data_2/data_2_fit.npy"

files = sorted(os.listdir(location))
#file=files[3]
file='data_2_xp.npy'
print(file)

reversed=False
Full_reversed = 1

n_avg=1
n_rolling = 1
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
    realtime=results[0]
    fs = round(1/np.median(realtime[1:]-realtime[:-1]))
    data_tot = results[1:]
    data_tot_avg=[]
    nevent=results.shape[-1]//2 #event_finder(results[1],fs)
    event_width=len(realtime) #int(6e-3*fs)
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
    realtime = realtime[::n_avg]
    while len(realtime)>data_tot_avg.shape[-1]:
        realtime=realtime[:-1]
    return(realtime,data_tot_avg)




realtime,data_tot = data_preparation(results,n_avg=n_avg,reversed=reversed)
fs = round(1/np.median(realtime[1:]-realtime[:-1]))

center = len(realtime)//2 #event_finder(data_tot[0],fs)
width=len(realtime)//2 #int(0.01*fs)
realtime=realtime[center-width:center+width]
data_tot=Full_reversed*data_tot[:,center-width:center+width]

### Simply plot data
#
# fig, axes = plt.subplots(3, 1,sharex=True)
#
# axes[0].plot(realtime,data_tot[0],label=r'$\varepsilon_{xx}$')
# axes[1].plot(realtime,data_tot[1],label=r'$\varepsilon_{yy}$')
# axes[2].plot(realtime,data_tot[2],label=r'$\varepsilon_{xy}$')
#
# axes[2].set_xlabel("time (s)")
# axes[1].set_ylabel('Strain')
#
#
# for ax in axes:
#     ax.grid(which='both')
#     ax.legend()
#
# plt.tight_layout()
# plt.show()
#


### Fitting function

with open("./DAQ_Python/full_epsilon.py") as f:
    exec(f.read())
    f.close()


def toopt(realtime,v,y_pos,t_0,Gamma):
    """
    Dummy function serving as an intermediary for scipy.curve_fit
    """
    args=v,y_pos,t_0,Gamma
    return(full_epsilon(realtime,args).reshape(-1))

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

args=curve_fit(toopt, realtime[start:end], data_tot[:,start:end].reshape(-1),bounds=bounds)
args=np.array((args[0],args[1][0],args[1][1]))

for i in range(len(args[0])):
    print(names[i],args[0,i])

print('\n\n')
### Compute results

xx,yy,xy = full_epsilon(realtime,args[0])
v,y_pos,t_0,Gamma=args[0]
x=-v*(realtime-t_0)

### Plot
fig, axes = plt.subplots(1, 3,sharex=True)

axes[0].plot(realtime,data_tot[0],label=r'$\varepsilon_{xx}$')
axes[1].plot(realtime,data_tot[1],label=r'$\varepsilon_{yy}$')
axes[2].plot(realtime,data_tot[2],label=r'$\varepsilon_{xy}$')

axes[0].plot(fit[0],fit[1],label=r"$Elsa's fit$")
axes[1].plot(fit[0],fit[2],label=r"$Elsa's fit$")
axes[2].plot(fit[0],fit[3],label=r"$Elsa's fit$")

line1,=axes[0].plot(realtime,xx,label=r'$\varepsilon_{xx}$ fit')
line2,=axes[1].plot(realtime,yy,label=r'$\varepsilon_{yy}$ fit')
line3,=axes[2].plot(realtime,xy,label=r'$\varepsilon_{xy}$ fit')

axes[2].set_xlabel("time (s)")
axes[1].set_ylabel('Strain')

for i in range(3):
    axes[i].axvline(realtime[end],linestyle="--",color='k')
    axes[i].axvline(realtime[start],linestyle="--",color='k')


for ax in axes:
    ax.grid(which='both')
    ax.legend()

plt.tight_layout()
#plt.show()


plt.subplots_adjust(bottom=0.6)


axfreq1 = plt.axes([0.25, 0.2, 0.65, 0.03])
v_slider = Slider(
    ax=axfreq1,
    label='v [m/s]',
    valmin=bounds[0,0],
    valmax=bounds[1,0],
    valinit=v,
)

axfreq2 = plt.axes([0.25, 0.3, 0.65, 0.03])
y_pos_slider = Slider(
    ax=axfreq2,
    label='y_pos [m]',
    valmin=bounds[0,1],
    valmax=bounds[1,1],
    valinit=y_pos,
)

axfreq3 = plt.axes([0.25, 0.4, 0.65, 0.03])
t_0_slider = Slider(
    ax=axfreq3,
    label='t_0 [s]',
    valmin=bounds[0,2],
    valmax=bounds[1,2],
    valinit=t_0,
)

axfreq4 = plt.axes([0.25, 0.5, 0.65, 0.03])
Gamma_slider = Slider(
    ax=axfreq4,
    label='$\Gamma$ [J/m]',
    valmin=bounds[0,3],
    valmax=bounds[1,3],
    valinit=Gamma,
)

def update(val):
    xx,yy,xy = full_epsilon(realtime,(v_slider.val, y_pos_slider.val,t_0_slider.val,Gamma_slider.val))
    line1.set_ydata(xx)
    line2.set_ydata(yy)
    line3.set_ydata(xy)
    fig.canvas.draw_idle()



v_slider.on_changed(update)
y_pos_slider.on_changed(update)
t_0_slider.on_changed(update)
Gamma_slider.on_changed(update)

#plt.xlim([-0.1,0.05])
plt.subplots_adjust(wspace=0.1)

plt.show()
args[0]=[v_slider.val,y_pos_slider.val,t_0_slider.val,Gamma_slider.val]

for i in range(len(args[0])):
    print(names[i],args[0,i])















