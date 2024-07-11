## Imports
# Science
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy import interpolate


try :
    from Python_DAQ import *
except :
    from DAQ_Python.Python_DAQ import *

#"plt.rcParams.update({"text.usetex": True})
#from matplotlib import rc
#rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
## read bin

location='D:/Users/Manips/Documents/DATA/FRICS/2022/2022-01-04/'
#location='D:/Users/Manips/Downloads/'
#location='F:/'
file = 'scope_2.bin'
fs = 1e9 #Sampling freq

time,data_tot= open_data_bin(location+file)
time=1000*time
data_tot=1000*data_tot

data_tot_filtered=np.copy(data_tot)



## Analysing
"""
from scipy.fft import fft, fftfreq
T=1/fs
N=len(data_tot[0])
yf=fft(data_tot[0])
xf=fftfreq(N,T)[:N//2]
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]),label='Unfiltered',alpha=0.7)
plt.grid("both")
plt.xscale("log")
plt.yscale("log")

plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (a.u.)")

plt.legend()
plt.show()

"""


## Cleaning

from scipy import signal

f0=[41500,44000,83000,123000,134000,164000,205000] #frequency to cut
Q = [10,10,10,10,10,10,10]
#quality
for i in range(len(f0)):
    b,a=signal.iirnotch(f0[i],Q[i],fs)
    for i in range(len(data_tot_filtered)):
        data_tot_filtered[i] = signal.filtfilt(b, a, data_tot_filtered[i])

"""
# lowpass
sos = signal.butter(4, 500, 'hp', fs=fs, output='sos')
for i in range(len(data_tot_filtered)):
    data_tot_filtered[i] = signal.sosfilt(sos, data_tot_filtered[i])

"""

## Seeing results of cleaning
"""
from scipy.fft import fft, fftfreq
T=1/fs
N=len(data_tot[0])
yf=fft(data_tot[0])
xf=fftfreq(N,T)[:N//2]
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]),label='Unfiltered',alpha=0.7)
plt.grid("both")
plt.xscale("log")
plt.yscale("log")


N=len(data_tot_filtered[0])
yf=fft(data_tot_filtered[0])
xf=fftfreq(N,T)[:N//2]
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]),label='filtered',alpha=0.7)
plt.grid()
plt.xscale("log")
plt.yscale("log")
plt.grid("both")

plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (a.u.)")

plt.legend()
#fig = plt.gcf()
#fig.set_size_inches(2.5,3)
#plt.tight_layout()
#plt.savefig("F:/spectrum.pdf")
#plt.close()
plt.show()
"""

## Average


n_avg=2000
data_tot_avg=[]
for i in range(len(data_tot)):
    data_tot_avg.append(avg_bin(data_tot_filtered[i],n_avg))
    data_tot_avg[i]=data_tot_avg[i]

data_tot_avg=np.array(data_tot_avg)

time_avg = time[::n_avg]

#scipy.io.savemat(location+file[:-4]+'.mat', dict(time_avg=time_avg*1000, data_1_avg=data_1_avg,data_2_avg=data_2_avg))

## plot


fig,axs=plt.subplots(len(data_tot),sharex=True)

start=0
colors=['r','g','b','orange','lime','grey','black','c','m','y']

for i in range(len(data_tot)):
    axs[i].plot(time_avg[start:],data_tot_avg[i][start:],label='Chan {}'.format(i+1), color=colors[i],linewidth=.5)


for i in range(len(data_tot)):
    axs[i].legend()
    axs[i].grid(which='both')


axs[-1].set(xlabel="Time (ms)")
axs[-1].set(ylabel="Voltage (mV)")
plt.show()







