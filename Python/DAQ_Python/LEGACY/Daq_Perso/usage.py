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
from Daq_Perso import *


## Constants
#ylabels=["Fn (kg)", "Fs (kg)", "d left (mm)", "d right (mm)"]

sampling_freq_in=10000
navg=100


## Live_Plot 2ch
ylabels=["Chan 1", "Chan 2"]

def calibration_functions():
    f_1 = lambda x:5*x+2
    f_2 = lambda x:np.log(np.abs(x))
    return(f_1, f_2)

data=live_plot( chans_in = 2,
                sampling_freq_in=sampling_freq_in,
                calibration_functions=calibration_functions,
                ylabels=ylabels,
                save=False,
                time_window=10,
                navg=navg,
                dev="Dev1")

## Plot old data
## Non live Plot n channels
speedup=1

loc="D:/Users/Manips/Documents/DATA/FRICS/2022/2022-09-30/daq_with_tilt.txt"
data=np.loadtxt(loc)


time=np.arange(len(data[0]))/sampling_freq_in*navg
fig, axs = plt.subplots(len(data),sharex=True)




for i in range(len(data)):
    axs[i].plot(time[::speedup],data[i][::speedup], label=ylabels[i])
    axs[i].grid("both")
    axs[i].legend()

axs[-1].set_xlabel('time (s)')

plt.show()
