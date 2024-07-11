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
file=files[0]
print(file)


time,data_tot= open_data_bin(location+file)
time=time[::5]
data_tot=data_tot[:,::5]
fs = round(1/np.median(time[1:]-time[:-1]))

data_tot_filtered=np.copy(data_tot)



## Analysing

from scipy.fft import fft, fftfreq
T=1/fs
N=len(data_tot[-1])
yf=fft(data_tot[-1])
xf=fftfreq(N,T)[:N//2]
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]),label='Unfiltered',alpha=0.7)
plt.grid("both")
plt.xscale("log")
plt.yscale("log")

plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (a.u.)")

plt.legend()
plt.show()


## Cleaning

"""
f0=[41500,44000,83000,123000,134000,164000,205000] #frequency to cut
Q = [10,10,10,10,10,10,10]
#quality
for i in range(len(f0)):
    b,a=signal.iirnotch(f0[i],Q[i],fs)
    for i in range(len(data_tot_filtered)):
        data_tot_filtered[i] = signal.filtfilt(b, a, data_tot_filtered[i])


# lowpass
sos = signal.butter(4, 500000, 'lp', fs=fs, output='sos')
for i in range(len(data_tot_filtered)):
    data_tot_filtered[i] = signal.sosfilt(sos, data_tot[i]-np.average(data_tot[i]))
    data_tot_filtered[i] = data_tot_filtered[i]


amp=np.std(data_tot_filtered[:,10000:],axis=-1)


data_tot_filtered=np.array([data_tot[i]-(amp[i]/amp[-1])*data_tot_filtered[-1] for i in range(3)])

"""





## Seeing results of cleaning
"""
from scipy.fft import fft, fftfreq
T=1/fs
N=len(data_tot_filtered[-1])
yf=fft(data_tot_filtered[-1])
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

n_avg=10

data_tot_avg=[]

for i in range(len(data_tot_filtered)):
    data_tot_avg.append(avg_bin(data_tot_filtered[i],n_avg))
    data_tot_avg[i]=data_tot_avg[i]

data_tot_avg=np.array(data_tot_avg)


"""
data_tot_med=[]
for i in range(len(data_tot)):
    data_tot_med.append(median_bin(data_tot_filtered[i],n_avg))
    data_tot_med[i]=data_tot_med[i]

data_tot_med=np.array(data_tot_med)
"""

time_avg = time[::n_avg]

while len(time_avg)>data_tot_avg.shape[-1]:
    print("lol")
    time_avg=time_avg[:-1]
#scipy.io.savemat(location+file[:-4]+'.mat', dict(time_avg=time_avg, data_1_avg=data_1_avg,data_2_avg=data_2_avg))

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





## transformations to restore tensor

def V_to_strain(data,amp=495,G=1.79,i_0=0.001712,R=350):
    return(data/(amp*R*G*i_0))



def rosette_to_tensor(ch_1,ch_2,ch_3):
    """
    converts a 45 degres rosette signal into a full tensor.
    input : the three channels of the rosette
    output : $\epsilon_{xx},\epsilon_{yy},\epsilon_{xy}
    https://www.efunda.com/formulae/solid_mechanics/mat_mechanics/strain_gage_rosette.cfm
    """
    eps_xx=ch_1+ch_3-ch_2
    eps_yy=ch_2
    eps_xy=(ch_1-ch_3)/2
    return(eps_xx,eps_yy,eps_xy)

## apply transformation

ch_1,ch_2,ch_3=V_to_strain(data_tot_avg[0],G=1.86),V_to_strain(data_tot_avg[1]),V_to_strain(data_tot_avg[2],G=1.86)
xx,yy,xy=rosette_to_tensor(ch_1,ch_2,ch_3)
tensor_epsilon=np.array([xx,yy,xy])


fig,axs=plt.subplots(len(tensor_epsilon),sharex=True)

start=0
colors=['r','g','b','orange','lime','grey','black','c','m','y']
lab=["$\epsilon_{xx}$","$\epsilon_{yy}$","$\epsilon_{xy}$"]


for i in range(len(tensor_epsilon)):
    axs[i].plot(time_avg[start:],tensor_epsilon[i][start:],label=lab[i], color=colors[i],linewidth=.5)


for i in range(len(tensor_epsilon)):
    axs[i].legend()
    axs[i].grid(which='both')


axs[-1].set(xlabel="Time (s)")
axs[-1].set(ylabel="Epsilon (Strain)")
plt.show()


## Save data

results=np.array([time_avg,xx,yy,xy])
np.save(location+file[:-4]+'_epsilon_time_xx_yy_xy.npy',results)


