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
ylabels=["Trigger (V)", "Position detector (mm)",  "chan 1","chan 2","chan 3","chan 4","chan 5", "chan 6"]

def calibration_temp():
    a,b,c=calibration_functions_2()
    positdet=lambda x: 2.3566*x-6.8572
    id=lambda x:x
    d=lambda x: V_to_strain(x,amp=495,G=1.79,i_0=0.0017,R=350)
    return(id,positdet,d,d,d,d,d,d)


data=live_plot(chans_in = 8, sampling_freq_in=sampling_freq_in,calibration_functions=calibration_temp, ylabels=ylabels, save=True,time_window=10,navg=navg,dev="Dev2",relative=[2,3,4,5,6,7])
