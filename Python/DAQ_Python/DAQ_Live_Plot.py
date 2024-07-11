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

sampling_freq_in=10000
navg=100



## Live_Plot 2ch
ylabels=["$F_n$ (kg)", "$F_f$ (kg)"]
data=live_plot(chans_in = 2, sampling_freq_in=sampling_freq_in,calibration_functions=calibration_functions_force, ylabels=ylabels, save=False,time_window=10,navg=navg,dev="Dev1")




## Live_Plot 3ch
ylabels=["$F_n$ (kg)", "$F_f$ (kg)", "Accelerometer(A.U.)"]#"ttl (V)"]
data=live_plot(chans_in = 3, sampling_freq_in=sampling_freq_in,calibration_functions=calibration_functions_2, ylabels=ylabels, save=False,time_window=5,navg=navg,dev="Dev1")

## Live_Plot 4ch
ylabels=["$F_n$ (kg)", "$F_f$ (kg)", "Accelerometer (A.U.)", "Position (mm)"]#"ttl (V)"]


def calibration_temp():
    a,b,c=calibration_functions_2()
    d=lambda x: 2.3566*x-6.8572
    return(d,a,b,c)



data=live_plot(chans_in = 4, sampling_freq_in=sampling_freq_in,calibration_functions=calibration_temp, ylabels=ylabels, save=False,time_window=5,navg=navg)




## Live_Plot 8ch
ylabels=["$F_n$ (kg)", "zz_1","zz_2","chan_2","chan_3","chan_4","chan_5","chan_6"]

i=1.7e-3

def calibration_temp():
    a,b,c=calibration_functions_force()
    d=lambda x: V_to_strain(x,amp=495,G=1.90,i_0=i,R=350)
    e=lambda x: V_to_strain(x,amp=495,G=1.86,i_0=i,R=350)
    f=lambda x: V_to_strain(x,amp=495,G=1.79,i_0=i,R=350)
    id=lambda x:x
    #return(a,id,id,id,id,id,id,id)
    return(a,d,d,e,f,e,e,f)



data=live_plot(chans_in = 8, sampling_freq_in=sampling_freq_in,calibration_functions=calibration_temp, ylabels=ylabels, save=True,time_window=10,navg=navg,dev="Dev1",relative=[1,2,3,4,5,6,7])




## Live_Plot 8ch
ylabels=["$F_n$ (kg)", "$F_f$ (kg)", "TTL", "chan 2","chan 5","chan 8","chan 11","chan 14"]

def calibration_temp():
    a,b,c=calibration_functions_force()
    d=lambda x: V_to_strain(x,amp=495,G=1.79,i_0=0.0017,R=350)
    return(a,b,c,d,d,d,d,a)



data=live_plot(chans_in = 8, sampling_freq_in=sampling_freq_in,calibration_functions=calibration_temp, ylabels=ylabels, save=True,time_window=10,navg=navg,dev="Dev2",relative=[3,4,5,6])


## Live Plot 6ch
ylabels=[ "chan 2","chan 5","chan 8","chan 11","chan 14","$F_n$ (kg)"]

def calibration_temp():
    a,b,c=calibration_functions_2()
    d=lambda x: V_to_strain(x,amp=495,G=1.79,i_0=0.0017,R=350)
    return(d,d,d,d,d,a)


data=live_plot(chans_in = 6, sampling_freq_in=sampling_freq_in,calibration_functions=calibration_temp, ylabels=ylabels, save=False,time_window=10,navg=navg,dev="Dev1",relative=[0,1,2,3,4])


## Live Plot 7ch
ylabels=["$F_n$ (kg)", "$F_s$ (Kg)", "$\epsilon_{yy}^{1}$","$\epsilon_{yy}^{2}$","$\epsilon_{yy}^{3}$","$\epsilon_{yy}^{4}$","$\epsilon_{yy}^{5}$"]

def calibration_temp():
    a,b,c=calibration_functions_force()
    alpha=lambda x:-a(x)
    d=lambda x: -1000*V_to_strain(x,amp=495,G=1.79,i_0=0.0017,R=350)
    return(alpha,alpha,d,d,d,d,d)


data=live_plot(chans_in = 7, sampling_freq_in=sampling_freq_in,calibration_functions=calibration_temp, ylabels=ylabels, save=True,time_window=10,navg=navg,dev="Dev1",relative=[2,3,4,5,6])



