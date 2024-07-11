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

plt.rcParams.update({"text.usetex": True})
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern Roman']})

## read bin


def open_data_bin_double(location):
    """
    Obsolete
    """
    from DAQ_Python.importAgilentBin import readfile
    time,data_1=readfile(location,0)
    _,data_2=readfile(location,1)
    return(time,data_1,data_2)

def open_data_bin(location):
    from DAQ_Python.importAgilentBin import readfile
    data_n=None
    data_tot=[]
    time=None
    bool_stop=False
    n=0
    while not bool_stop:
        time_temp,data_n=readfile(location,n)
        n+=1
        bool_stop= data_n is None
        if not bool_stop:
            data_tot.append(data_n)
            time=time_temp
    return(time,data_tot)


### data_anal

location='D:/Users/Manips/Documents/DATA/FRICS/2021-07-23/'
file = 'scope_{}.bin'

avgs=[]
stds=[]
meds=[]
for i in range(48):
    time,data_tot= open_data_bin(location+file.format(i))
    time=np.array(time)
    data_tot=np.array(data_tot)
    avgs.append(np.mean(data_tot))
    stds.append(np.std(data_tot))
    meds.append(np.median(data_tot))


for a in avgs:
    print(a)

for a in stds:
    print(a)

for a in meds:
    print(a)

