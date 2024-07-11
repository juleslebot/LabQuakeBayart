
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy import signal

from scipy import interpolate
try :
    from Python_DAQ import *
except :
    from DAQ_Python.Python_DAQ import *



def event_finder(signal,fs):
    n_smooth=int(0.00001*fs)
    a=np.diff(avg_bin(signal,n_smooth))
    #plt.plot(a)
    #plt.show()
    return(n_smooth*np.argmax(np.abs(a)))

chan=2

location='D:/Users/Manips/Documents/DATA/FRICS/2022/2022-05-25/xp1/npy_raw_u{}/'.format(chan)
n_avg=5
roll_avg=200
permutation = [1,0,2,3]
label=['ch 5', 'ch 2', 'ch 8', 'ch 14']


for i in range(0,5):
    print(i)
    file1 = 'osc_1_{}_time_u{}1_u{}2.npy'.format(i,chan,chan)
    file2 = 'osc_2_{}_time_u{}1_u{}2.npy'.format(i,chan,chan)

    a=np.load(location+file1)
    time1,data_tot1= a[0],a[1:]


    data_tot1_avg=[]
    for i in range(len(data_tot1)):
        data_tot1_avg.append(avg_bin(data_tot1[i],n_avg))
    data_tot1=np.array(data_tot1_avg)
    time1 = time1[::n_avg]



    #time=time-time[len(time)//2]
    while len(time1)>data_tot1.shape[-1]:
        time1=time1[:-1]
    fs1 = round(1/np.median(time1[1:]-time1[:-1]))

    data_tot1=smooth(data_tot1,roll_avg)
    time1=smooth(time1,roll_avg)

    a=np.load(location+file2)
    time2,data_tot2= a[0],a[1:]

    data_tot2_avg=[]
    for i in range(len(data_tot2)):
        data_tot2_avg.append(avg_bin(data_tot2[i],n_avg))
    data_tot2=np.array(data_tot2_avg)
    time2 = time2[::n_avg]
    while len(time2)>data_tot2.shape[-1]:
        time2=time2[:-1]
    fs2 = round(1/np.median(time2[1:]-time2[:-1]))

    data_tot2=smooth(data_tot2,roll_avg)
    time2=smooth(time2,roll_avg)


    #t1=event_finder(data_tot1[composante],fs1)
    #t2=event_finder(data_tot2[composante],fs2)
    #print("c = ",7.6e-2/(time2[t2]-time1[t1]))

    fig, axes = plt.subplots(4, 1,sharex=True)



    axes[permutation[0]].plot(time1,data_tot1[0],label=label[0])
    axes[permutation[1]].plot(time1,data_tot1[1],label=label[1])
    axes[permutation[2]].plot(time2,data_tot2[0],label=label[2])
    axes[permutation[3]].plot(time2,data_tot2[1],label=label[3])

    axes[3].set_xlabel("time (s)")
    axes[3].set_ylabel('Strain')

    #axes[0].axvline(time1[t1],linestyle="--",color='k')
    #axes[1].axvline(time2[t2],linestyle="--",color='k')

    for ax in axes:
        ax.grid(which='both')
        ax.legend()

    plt.tight_layout()
    plt.show()
