## Imports
# Science
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy import interpolate
import time as TIME

# DAQ
import nidaqmx
from nidaqmx.stream_readers import AnalogMultiChannelReader
from nidaqmx import constants
# random
import threading
from threading import Thread
import pickle
try :
    from Python_DAQ import *
except :
    from DAQ_Python.Python_DAQ import *


## Constants
#ylabels=["Fn (kg)", "Fs (kg)", "d left (mm)", "d right (mm)"]

sampling_freq_in=10000
navg=1
location_temp="D:/Users/Manips/Downloads/"
location_final="D:/Users/Manips/Downloads/"

ylabels_1=["chan 1","chan 2","chan 3","chan 4","chan 5", "chan 6", "chan 7","chan 8"]
ylabels_2=["chan 9", "chan 10","chan 11","chan 12","chan 13","chan 14", "chan 15","$F_s$ (kg)"]

## Live_Plot 16ch


def func1():
    def calibration_temp():
        a,b,c=calibration_functions_force()
        d=lambda x: -V_to_strain(x,amp=495,G=1.79,i_0=0.0017,R=350)
        e=lambda x: -V_to_strain(x,amp=495,G=1.86,i_0=0.0017,R=350)
        return(e,d,e,e,d,e,e,d)

    data=live_plot(chans_in = 8, sampling_freq_in=sampling_freq_in, calibration_functions=calibration_temp, ylabels=ylabels_1, save=location_temp + "temp1.npy", navg=navg, dev="Dev1", relative=[0,1,2,3,4,5,6,7], no_plot=True,buffer_in_size=100000)


def func2():
    def calibration_temp():
        a,b,c=calibration_functions_force()
        d=lambda x: -V_to_strain(x,amp=495,G=1.79,i_0=0.0017,R=350)
        e=lambda x: -V_to_strain(x,amp=495,G=1.86,i_0=0.0017,R=350)
        f=lambda x: -a(x)
        return(e,e,d,e,e,d,e,f)



    data=live_plot2(chans_in = 8, sampling_freq_in=sampling_freq_in, calibration_functions=calibration_temp,  ylabels=ylabels_2, save=location_temp + "temp2.npy", navg=navg, dev="Dev2", relative=[0,1,2,3,4,5,6], no_plot=True,buffer_in_size=100000)




thread1=Thread(target = func2)
thread2=Thread(target = func1)

thread1.start()
thread2.start()


## Recombine the two files

data_1=np.load(location_temp+"temp1.npy")
data_2=np.load(location_temp+"temp2.npy")
data=concatarrays(data_1,data_2)
ylabels=ylabels_1+ylabels_2
permutation=[]
daq={"data":data,"ylabels":ylabels,"sampling_freq_in":sampling_freq_in,"navg":navg}
np.save(location_final+"test.npy",daq,allow_pickle=True)



## Recombine with exterior element, step 1

loc_out='D:/Users/Manips/Downloads/scope_20.bin'
n_avg=1

time_out,data_out=open_data_bin(loc_out)
time=np.arange(len(data[0]))/sampling_freq_in*navg

def cal_func_Fn(x):
    G=500/3
    return(G*x)

data_out=-cal_func_Fn(rolling_average(data_out[0],n_avg))
time_out=rolling_average(time_out,n_avg)
time_out=time_out-time_out[0]
time_out=time_out+time[-1]-time_out[-1]

fig, axs = plt.subplots(2,sharex=True)
axs[0].plot(time,data[-1])
axs[1].plot(time_out,data_out)
plt.show()

## Recombine with exterior element, step 2

t_event_1=5.9739
t_event_2=3.5166

time_out=time_out+t_event_1-t_event_2

#fig, axs = plt.subplots(2,sharex=True)
#axs[0].plot(time,data[-1])
#axs[1].plot(time_out,data_out)
#plt.show()


###

new_data=retime_data(time,time_out,data_out)

data=np.concatenate([data,[new_data]])
ylabels=ylabels+["$F_n$ (Kg)"]

daq={"data":data,"ylabels":ylabels,"sampling_freq_in":sampling_freq_in,"navg":navg}

daq["data"][-1][daq["data"][-1]<-10]=0
daq["data"][-2][daq["data"][-2]<-10]=0


np.save(location_final+"Full_Daq.npy",daq,allow_pickle=True)












## Quick verification plot dev 1

def calib_verif():
    a,b,c=calibration_functions_2()
    id=lambda x:x
    return(a)

data=live_plot(chans_in = 1, sampling_freq_in=sampling_freq_in, calibration_functions=calib_verif, ylabels=["Force (kg)"], save=False, navg=navg, dev="Dev1")

## Quick verification plot dev 2

def calib_verif():
    a,b,c=calibration_functions_2()
    id=lambda x:x
    return(a)

data=live_plot(chans_in = 1, sampling_freq_in=sampling_freq_in, calibration_functions=calib_verif, ylabels=[ "Force (kg)"], save=False, navg=navg, dev="Dev2")


























###

ylabels=["chan 1","chan 2","chan 3","chan 4","chan 5","chan 6","chan 7","chan 8"]

sampling_freq_in=1000
navg=100
def calibration_temp():
    a,b,c=calibration_functions_force()
    d=lambda x: V_to_strain(x,amp=495,G=1.79,i_0=0.0017,R=350)
    e=lambda x: V_to_strain(x,amp=495,G=1.86,i_0=0.0017,R=350)
    return(e,d,e,e,d,e,e,d)



data=live_plot(chans_in = 8, sampling_freq_in=sampling_freq_in,calibration_functions=calibration_temp, ylabels=ylabels, save=True,time_window=10,navg=navg,dev="Dev2",relative=[0,1,2,3,4,5,6,7])

###

ylabels=["chan 9","chan 10","chan 11","chan 12","chan 13","chan 14","chan 15","Fs"]

sampling_freq_in=1000
navg=100
def calibration_temp():
    a,b,c=calibration_functions_force()
    d=lambda x: V_to_strain(x,amp=495,G=1.79,i_0=0.0017,R=350)
    e=lambda x: V_to_strain(x,amp=495,G=1.86,i_0=0.0017,R=350)
    return(e,e,d,e,e,d,e,a)


data=live_plot(chans_in = 8, sampling_freq_in=sampling_freq_in,calibration_functions=calibration_temp, ylabels=ylabels, save=True,time_window=10,navg=navg,dev="Dev1",relative=[0,1,2,3,4,5,6])

