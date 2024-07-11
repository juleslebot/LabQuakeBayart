## Imports
# Science
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy import signal
import os

from scipy import interpolate
try :
    from Python_DAQ import *
except :
    from DAQ_Python.Python_DAQ import *


## read bin

location='D:/Users/Manips/Documents/DATA/FRICS/2022/2022-02-18/bin/'
#location='D:/Users/Manips/Downloads/'
#location='F:/'

files = sorted(os.listdir(location))

n_avg=20
n_rolling = 20

time1,data_tot1= open_data_bin(location+files[3])
time1=smooth(time1,n_rolling)
data_tot1=smooth(data_tot1,n_rolling)

def transformation1(time1, data_tot1,n_avg=1, delay = 0):
    data_tot=np.copy(data_tot1)
    time=np.copy(time1)
    fs = round(1/np.median(np.diff(time)))
    n_decal = int(delay * fs)
    print(n_decal)
    if n_decal!=0 :
        time=time[n_decal:-n_decal]

        ch1,ch2,ch3=data_tot[0],data_tot[1],data_tot[2]

        ch1=ch1[:-n_decal*2]
        ch2=ch2[n_decal:-n_decal]
        ch3=ch3[n_decal*2:]

        data_tot=np.array([ch1,ch2,ch3])
    elif delay !=0:
        print("No delay applied")
    time=time[10000-n_decal:-10000+n_decal]
    data_tot=data_tot[:,10000-n_decal:-10000+n_decal]
    data_tot_filtered=np.copy(data_tot)

    data_tot_avg=[]

    for i in range(len(data_tot_filtered)):
        data_tot_avg.append(avg_bin(data_tot_filtered[i],n_avg))
        data_tot_avg[i]=data_tot_avg[i]

    data_tot_avg=np.array(data_tot_avg)

    time_avg = time[::n_avg]

    while len(time_avg)>data_tot_avg.shape[-1]:
        time_avg=time_avg[:-1]

    ch_1,ch_2,ch_3=V_to_strain(data_tot_avg[0],G=1.86),V_to_strain(data_tot_avg[1]),V_to_strain(data_tot_avg[2],G=1.86)
    xx,yy,xy=rosette_to_tensor(ch_1,ch_2,ch_3)

    return(time_avg,xx,yy,xy)





time,xx,yy,xy=transformation1(time1,data_tot1,n_avg=n_avg,delay=0)



##
from matplotlib.widgets import Slider, Button


#set bounds
bounds=[ 0     ,2000  ] #v
v=1000
# set proportion of the data to take into account



time,xx,yy,xy = transformation1(time1,data_tot1,n_avg=n_avg,delay=0.5e-3/v)



### Plot
fig, axes = plt.subplots(1, 3,sharex=True)

line1,=axes[0].plot(time,xx)
line2,=axes[1].plot(time,yy)
line3,=axes[2].plot(time,xy)

axes[1].set_xlabel("time (s)")
axes[0].set_ylabel(r'$\varepsilon_{xx}$')
axes[1].set_ylabel(r'$\varepsilon_{yy}$')
axes[2].set_ylabel(r'$\varepsilon_{xy}$')



for ax in axes:
    ax.grid(which='both')


plt.tight_layout()
#plt.show()


plt.subplots_adjust(bottom=0.6)


axfreq1 = plt.axes([0.25, 0.2, 0.65, 0.03])
v_slider = Slider(
    ax=axfreq1,
    label='v [m/s]',
    valmin=bounds[0],
    valmax=bounds[1],
    valinit=v,
)

def update(val):
    time,xx,yy,xy = transformation1(time1,data_tot1,n_avg=n_avg,delay=0.5e-3/v_slider.val)
    line1.set_ydata(xx)
    line2.set_ydata(yy)
    line3.set_ydata(xy)
    fig.canvas.draw_idle()



v_slider.on_changed(update)

#plt.xlim([-0.1,0.05])
plt.subplots_adjust(wspace=0.1)

plt.show()

















## read bin

location='D:/Users/Manips/Documents/DATA/FRICS/2022/2022-02-18/bin/'
#location='D:/Users/Manips/Downloads/'
#location='F:/'

files = sorted(os.listdir(location))

n_avg=20
n_rolling = 20

time1,data_tot1= open_data_bin(location+files[3])
time1=smooth(time1,n_rolling)
fs = round(1/np.median(np.diff(time1)))
data_tot1=smooth(data_tot1,n_rolling)

def transformation1(time1, data_tot1,n_avg=1, n_decal = 0):
    data_tot=np.copy(data_tot1)
    time=np.copy(time1)
    fs = round(1/np.median(np.diff(time)))
    time=time[10000:-10000]
    ch1,ch2,ch3=data_tot[0],data_tot[1],data_tot[2]
    ch2=ch2[10000:-10000]
    ch1=ch1[10000-n_decal:-10000-n_decal]
    ch3=ch3[10000+n_decal:-10000+n_decal]
    data_tot=np.array([ch1,ch2,ch3])
    data_tot_filtered=np.copy(data_tot)

    data_tot_avg=[]

    for i in range(len(data_tot_filtered)):
        data_tot_avg.append(avg_bin(data_tot_filtered[i],n_avg))
        data_tot_avg[i]=data_tot_avg[i]

    data_tot_avg=np.array(data_tot_avg)

    time_avg = time[::n_avg]

    while len(time_avg)>data_tot_avg.shape[-1]:
        time_avg=time_avg[:-1]

    ch_1,ch_2,ch_3=V_to_strain(data_tot_avg[0],G=1.86),V_to_strain(data_tot_avg[1]),V_to_strain(data_tot_avg[2],G=1.86)
    xx,yy,xy=rosette_to_tensor(ch_1,ch_2,ch_3)

    return(time_avg,xx,yy,xy)



time,xx,yy,xy=transformation1(time1,data_tot1,n_avg=n_avg,n_decal=0)



##
from matplotlib.widgets import Slider, Button


#set bounds
bounds=[ -40,40 ] #v
v=0
# set proportion of the data to take into account



time,xx,yy,xy = transformation1(time1,data_tot1,n_avg=n_avg,n_decal=0)



### Plot
fig, axes = plt.subplots(1, 3,sharex=True)

line1,=axes[0].plot(time,xx)
line2,=axes[1].plot(time,yy)
line3,=axes[2].plot(time,xy)

axes[1].set_xlabel("time (s)")
axes[0].set_ylabel(r'$\varepsilon_{xx}$')
axes[1].set_ylabel(r'$\varepsilon_{yy}$')
axes[2].set_ylabel(r'$\varepsilon_{xy}$')



for ax in axes:
    ax.grid(which='both')


plt.tight_layout()
#plt.show()


plt.subplots_adjust(bottom=0.6)


axfreq1 = plt.axes([0.25, 0.2, 0.65, 0.03])
v_slider = Slider(
    ax=axfreq1,
    label='n_decal',
    valmin=bounds[0],
    valmax=bounds[1],
    valinit=v,
    valstep=1
)

def update(val):
    time,xx,yy,xy = transformation1(time1,data_tot1,n_avg=n_avg,n_decal=int(v_slider.val))
    line1.set_ydata(xx)
    line2.set_ydata(yy)
    line3.set_ydata(xy)
    try:
        print("estimated speed :",5e-4*fs/abs(v_slider.val))
    except:
        pass
    fig.canvas.draw_idle()



v_slider.on_changed(update)

#plt.xlim([-0.1,0.05])
plt.subplots_adjust(wspace=0.1)

plt.show()




