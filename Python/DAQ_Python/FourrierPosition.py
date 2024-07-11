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


## read bin

location='D:/Users/Manips/Documents/DATA/FRICS/2021-05-31/resultsAnalysis/'

file = 'bruit.txt'

def open_data_txt(location):
    op=open(location)
    data_1=op.read()
    data_1='['+data_1[:-1]+']'
    data_1=eval(data_1)
    op.close()
    return(data_1)

data_1=open_data_txt(location+file)


## Cleaning
data_1_filtered=np.copy(data_1)

from scipy import signal
f_0 = 2e4 #frequency to cut
fs = 2.3e5 #Sampling freq
order = 1
b,a=signal.butter(order,f_0/fs,btype='low',analog=False)
data_1_filtered = signal.filtfilt(b, a, data_1_filtered)

## Analysing
from scipy.fft import fft, fftfreq
T=1/(4e9)
N=len(data_1)
yf=fft(data_1)
xf=fftfreq(N,T)[:N//2]
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.grid()
plt.xscale("log")
plt.yscale("log")

N=len(data_1_filtered)
yf=fft(data_1_filtered)
xf=fftfreq(N,T)[:N//2]
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.grid()
plt.xscale("log")
plt.yscale("log")
plt.grid("both")

plt.show()


## plot

plt.plot(data_1)
plt.grid(which='both')
plt.show()
