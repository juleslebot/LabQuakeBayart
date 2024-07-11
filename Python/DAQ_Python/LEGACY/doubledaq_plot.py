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



## 16 channels dictionnary Plot
loc="D:/Users/Manips/Documents/DATA/FRICS/2022/2022-09-23/daq_0.npy"
loc="D:/Users/Manips/Downloads/test.npy"
speedup=10





data_dict=np.load(loc,allow_pickle=True).all()
ylabels=data_dict["ylabels"]
data=data_dict["data"]
sampling_freq_in=data_dict["sampling_freq_in"]
navg=data_dict["navg"]

time=np.arange(len(data[0]))/sampling_freq_in*navg




fig, axs = plt.subplots(8,2,sharex=True)

for i in range(len(data)):
    axs[i%8][i//8].plot(time[::speedup],data[i][::speedup], label=ylabels[i])
    axs[i%8][i//8].grid("both")
    axs[i%8][i//8].legend()


axs[-1][0].set_xlabel('time (s)')

plt.show()






## Constants
#ylabels=["Fn (kg)", "Fs (kg)", "d left (mm)", "d right (mm)"]

sampling_freq_in=1000
navg=10



## Non live Plot 16 channels
speedup=1
fig, axs = plt.subplots(8,2,sharex=True)


ylabels=["Trigger (V)", "$F_n$ (kg)",  "chan 1","chan 2","chan 3","chan 4","chan 5", "chan 6"]


loc="D:/Users/Manips/Documents/DATA/FRICS/2022/2022-09-21/daq1_1-6.txt"
loc="D:/Users/Manips/Downloads/temp1.npy"
#loc="D:/Users/Manips/Documents/DATA/FRICS/2022/2022-08-02-Portrait de Phase/manips/xp53-retour.txt"
data=np.load(loc)

time=np.arange(len(data[0]))/sampling_freq_in*navg




for i in range(len(data)):
    axs[i][0].plot(time[::speedup],data[i][::speedup], label=ylabels[i])
    axs[i][0].grid("both")
    axs[i][0].legend()




ylabels=["Trigger (V)", "$F_f$ (kg)", "chan 10","chan 11","chan 12","chan 13","chan 14", "chan 15"]

loc="D:/Users/Manips/Documents/DATA/FRICS/2022/2022-09-21/daq1_10-15.txt"
loc="D:/Users/Manips/Downloads/temp2.npy"

#loc="D:/Users/Manips/Documents/DATA/FRICS/2022/2022-08-02-Portrait de Phase/manips/xp53-retour.txt"
data2=np.load(loc)

time2=np.arange(len(data2[0]))/sampling_freq_in*navg




for i in range(len(data2)):
    axs[i][1].plot(time2[::speedup],data2[i][::speedup], label=ylabels[i])
    axs[i][1].grid("both")
    axs[i][1].legend()

axs[-1][0].set_xlabel('time (s)')


#fig.suptitle(loc[31:])




#plt.savefig("C:/Users/Manips/test.png")
#plt.close()
plt.show()