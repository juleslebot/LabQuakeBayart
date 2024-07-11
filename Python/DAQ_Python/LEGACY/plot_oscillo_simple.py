## Imports
# Science
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy import signal

from scipy import interpolate
try :
    from Python_DAQ import *
except :
    from DAQ_Python.Python_DAQ import *

import os

#"plt.rcParams.update({"text.usetex": True})
#from matplotlib import rc
#rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
## read bin

location='D:/Users/Manips/Documents/DATA/FRICS/2022/2022-05-10/bin/'
#location='D:/Users/Manips/Downloads/'
#location='F:/'

files = sorted(os.listdir(location))
file=files[3]
print(file)


time,data_tot= open_data_bin(location+file)
fs = round(1/np.median(time[1:]-time[:-1]))

data_tot_filtered=np.copy(data_tot)



## Average

n_avg=1

roll_avg=1

data_tot_avg=[]

for i in range(len(data_tot_filtered)):
    data_tot_avg.append(avg_bin(data_tot_filtered[i],n_avg))
    data_tot_avg[i]=data_tot_avg[i]



data_tot_avg=np.array(data_tot_avg)
time_avg = time[::n_avg]

while len(time_avg)>data_tot_avg.shape[-1]:
    print("lol")
    time_avg=time_avg[:-1]
#scipy.io.savemat(location+file[:-4]+'.mat', dict(time_avg=time_avg, data_1_avg=data_1_avg,data_2_avg=data_2_avg))
data_tot_avg=smooth(data_tot_avg,roll_avg)
time_avg=smooth(time_avg,roll_avg)



## plot

fig,axs=plt.subplots(len(data_tot),sharex=True)

start=0
colors=['r','g','b','orange','lime','grey','black','c','m','y']

for i in range(len(data_tot)):
    axs[i].plot(time[start:],data_tot[i][start:],label='Chan {}'.format(i+1), color=colors[i],linewidth=.5,alpha=0.2)
    axs[i].plot(time_avg[start:],data_tot_avg[i][start:],label='Chan {}'.format(i+1), color=colors[i],linewidth=.5)


for i in range(len(data_tot)):
    axs[i].legend()
    axs[i].grid(which='both')


axs[-1].set(xlabel="Time (s)")
axs[-1].set(ylabel="Voltage (V)")
plt.show()

