## Imports
# Science
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy import interpolate
# DAQ
import nidaqmx
from nidaqmx.stream_readers import AnalogMultiChannelReader
from nidaqmx import constants
# random
import threading
import pickle
try :
    from Python_DAQ import *
except :
    from DAQ_Python.Python_DAQ import *


## Constants
#ylabels=["Fn (kg)", "Fs (kg)", "d left (mm)", "d right (mm)"]

sampling_freq_in=100000
navg=100



## Non live Plot n channels
speedup=1
#sampling_freq_in=10000

#ylabels=["$F_n$ (kg)", "$F_f$ (kg)", "Accelerometer(A.U.)", "Position (mm)"]#"Trigger (V)"]
#ylabels=["$F_n$ (kg)", "$F_f$ (kg)", "TTL", "chan 2","chan 5","chan 8","chan 11","chan 14"]

loc="D:/Users/Manips/Documents/DATA/FRICS/2022/2022-11-22/daq4.txt"
#loc="D:/Users/Manips/Documents/DATA/FRICS/2022/2022-08-02-Portrait de Phase/manips/xp53-retour.txt"
data=np.loadtxt(loc)

time=np.arange(len(data[0]))/sampling_freq_in*navg
fig, axs = plt.subplots(len(data),sharex=True)




for i in range(len(data)):
    axs[i].plot(time[::speedup],data[i][::speedup], label=ylabels[i])
    axs[i].grid("both")
    axs[i].legend()

axs[-1].set_xlabel('time (s)')
#fig.suptitle(loc[31:])
#plt.tight_layout()
#fig.set_size_inches((9,7))

#plt.savefig("C:/Users/Manips/test.png")
#plt.close()
plt.show()

## Non live Plot 8ch
speedup=1 #used to speed up the ploting process by decimating the data

ylabels=["$F_n$ (kg)", "$F_f$ (kg)", "TTL", "chan 2","chan 5","chan 8","chan 11","chan 14"]

loc="D:/Users/Manips/Documents/DATA/FRICS/2022/2022-10-11/temps de chauffe.txt"
data=np.loadtxt(loc)

time=np.arange(len(data[0]))/sampling_freq_in*navg/60/10
fig, axs = plt.subplots(3)
axs[0].plot(time[::speedup],data[0][::speedup], label=ylabels[0])
axs[1].plot(time[::speedup],data[1][::speedup], label=ylabels[1])
axs[0].grid("both")
axs[0].legend()
axs[1].grid("both")
axs[1].legend()

for i in range(3,8):
    axs[2].plot(time[::speedup],1000*(data[i][::speedup]-data[i][0:10].mean()+i/1000000), label=ylabels[i])
    axs[2].grid("both")


axs[2].set_xlabel('time (mn)')
axs[2].set_ylabel('mStrains')
axs[2].legend()
plt.show()


## Non live Plot 3ch
speedup=1
sampling_freq_in=1000
ylabels=["$F_n$ (kg)", "$F_f$ (kg)", "TTL"]

loc="D:/Users/Manips/Documents/DATA/FRICS/2022/2022-06-29/daq2_cam.txt"
data=np.loadtxt(loc)

time=np.arange(len(data[0]))/(100*60)
fig, axs = plt.subplots(3)
axs[0].plot(time[::speedup],data[0][::speedup], label=ylabels[0])
axs[1].plot(time[::speedup],data[1][::speedup], label=ylabels[1])
axs[0].grid("both")
axs[0].legend()
axs[1].grid("both")
axs[1].legend()
axs[2].plot(time[::speedup],data[2][::speedup], label=ylabels[2])
axs[2].grid("both")
axs[2].set_xlabel('time (mn)')
axs[2].set_ylabel('TTL (V)')
axs[2].legend()
fig.suptitle(loc)
plt.show()




## Non-live plotting 3ch with friction coeff
speedup=1
sampling_freq_in=1000
ylabels=["$F_n$ (kg)", "$F_f$ (kg)", "Trigger (V)","$\mu=F_f\,/F_n$"]

loc="D:/Users/Manips/Documents/DATA/FRICS/2022/2022-06-29/daq2_cam.txt"
data=np.loadtxt(loc)

time=np.arange(len(data[0]))/(100*60)
fig, axs = plt.subplots(4)

databis=np.vstack([data,data[1]/data[0]])



for i in range(4):
    axs[i].plot(time[::speedup],databis[i][::speedup], label=ylabels[i])
    axs[i].grid("both")
    axs[i].legend()

axs[-1].set_xlabel('time (mn)')
fig.suptitle(loc)
plt.show()


## Non-live plotting 3ch with friction coeff
sampling_freq_in=1000
ylabels=["$F_n$ (kg)", "$F_f$ (kg)", "Trigger (V)","F_f/F_n"]

loc="D:/Users/Manips/Documents/DATA/FRICS/2022/2022-06-30/daq2.txt"
data=np.loadtxt(loc)
databis=np.vstack([data,data[1]/data[0]])
quick_plot_data(databis,1000,xlabel="Time (s)",ylabels=ylabels)
fig = plt.gcf()
fig.set_size_inches(3.937,3)
plt.tight_layout()
#plt.savefig("D:/Users/Manips/Documents/DATA/FRICS/2021/2021-11-22/acq_2/acq_2.pdf")
plt.show()

#plt.close()


## Plot

sampling_freq_in=1000
ylabels=["$F_n$ (kg)", "$F_f$ (kg)", "TTL", "chan 2","chan 5","chan 8","chan 11","chan 14"]

loc="D:/Users/Manips/Documents/DATA/FRICS/2022/2022-05-25/load_unload_granular_1cmgrain.txt"
data=np.loadtxt(loc)



for i in range(3,8):
    plt.scatter(data[0],0.0001*i+data[i]-np.min(data[i]),label="chan {}".format(3*i-7),alpha=.5,s=5)

plt.legend()
plt.show()







##Calibration position detector

def calibration_position_detector(file="D:/Users/Manips/Documents/DATA/FRICS/2021/calibration_position.txt",sampling_freq_in=sampling_freq_in,speed=1):#mm/s
    data=np.loadtxt(file)
    data=data[-1,:]
    data_bis=data[data>0.9]
    time=np.arange(0,len(data_bis))/sampling_freq_in
    fit = np.polyfit(time, data_bis, 1)
    # plt.plot(time,data_bis)
    # plt.plot(time,time*fit[0]+fit[1])
    # plt.show()
    return(fit[0]/speed)


##Live Plot with position captor
ylabels=["$F_n$ (kg)", "$F_f$ (kg)", "ttl (V)", "position (mm)"]

calib=calibration_position_detector(speed=1)

def calibration_temp():
    a,b,c=calibration_functions_2()
    d=lambda x: x/calib
    return(a,b,c,d)



data=live_plot(chans_in = 4, sampling_freq_in=sampling_freq_in,calibration_functions=calibration_temp, ylabels=ylabels, save=True,time_window=5,navg=1)








## Live Plot 4ch
ylabels=["F1", "F2", "trig", "cam"]

data=live_plot(chans_in = 4, sampling_freq_in=sampling_freq_in, ylabels=ylabels, save=False,time_window=5)

##
data=live_plot(chans_in = 4, sampling_freq_in=sampling_freq_in, time_window=15)

## Live Plot D63 only

data=live_plot(chans_in = 1, sampling_freq_in=sampling_freq_in, time_window=15,navg=100, buffer_in_size=1000)#,calibration_functions=calibration_function_D63_short_range, ylabels="Distance (mm)", save=True, max_D63=2.2195)

## Live Plot Accel only

data=live_plot(chans_in = 1, sampling_freq_in=sampling_freq_in, time_window=15,navg=10, buffer_in_size=1000, ylabels="Accel (u.a.)", save=False)



## Non-live plotting
sampling_freq_in=1000
ylabels=["$F_n$ (kg)", "$F_f$ (kg)", "Trigger (V)"]#, 'position (mm)']

loc="D:/Users/Manips/Documents/DATA/FRICS/2022/2022-06-29/daq2.txt"
data=np.loadtxt(loc)
quick_plot_data(data,1000,xlabel="Time (s)",ylabels=ylabels)
fig = plt.gcf()
fig.set_size_inches(3.937,3)
plt.tight_layout()
#plt.savefig("D:/Users/Manips/Documents/DATA/FRICS/2021/2021-11-22/acq_2/acq_2.pdf")
plt.show()

#plt.close()



















##


for i in range(8,9):
    filename="Full_daq_{}".format(i)
    data=np.loadtxt(loc+filename+".txt")
    data[1,:]=-data[1,:]
    if np.average(data[1,:])<0:
        data[1,:]=-data[1,:]
        data[3,:]=-data[3,:]
    quick_plot_data(data[:3,100000:498000],sampling_freq_in,ylabels=ylabels)
    x=np.arange(data[3,100000:498000].size)/sampling_freq_in
    fig=plt.gcf()
    fig.set_size_inches(5,4)
    ax=fig.axes[1]
    ax2=ax.twinx()
    ax2.plot(x,-data[3,100000:498000],color="orange")
    ax2.set_ylim(9.35,10.35)
    plt.tight_layout()
    plt.savefig(loc+filename+".pdf")
    plt.close()

##
quick_plot_data(data,1000,xlabel="Time (s)",ylabels=ylabels)

fig = plt.gcf()
fig.set_size_inches(3.937,3)
plt.tight_layout()
plt.savefig("F:/stick_slip_pmma.pdf")




#D:/Users/Manips/Documents/DATA/FRICS/2021-03-22/crack