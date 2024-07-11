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

sampling_freq_in=1000
navg=10





## Live_Plot 8ch
ylabels=["$F_n$ (kg)", "$F_f$ (kg)", "chan 10","chan 11","chan 12","chan 13","chan 14", "chan 15"]

def calibration_temp():
    a,b,c=calibration_functions_2()
    d=lambda x: V_to_strain(x,amp=495,G=1.79,i_0=0.0017,R=350)
    return(a,b,d,d,d,d,d,d)


data=live_plot(chans_in = 8, sampling_freq_in=sampling_freq_in,calibration_functions=calibration_temp, ylabels=ylabels, save="D:/Users/Manips/Downloads/test2.txt" ,time_window=10,navg=navg,dev="Dev1",relative=[2,3,4,5,6,7],no_plot=True)
