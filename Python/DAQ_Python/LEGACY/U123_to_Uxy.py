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

## read bin

location='D:/Users/Manips/Documents/DATA/FRICS/2022/2022-06-24/bins/'
#location='D:/Users/Manips/Downloads/'
#location='F:/'
chan=2
files = sorted(os.listdir(location))

n_avg=1

def transformation1(location,n_avg=1):

    time,data_tot= open_data_bin(location)
    fs = round(1/np.median(time[1:]-time[:-1]))

    data_tot_filtered=np.copy(data_tot)

    data_tot_avg=[]

    for i in range(len(data_tot_filtered)):
        data_tot_avg.append(avg_bin(data_tot_filtered[i],n_avg))
        data_tot_avg[i]=data_tot_avg[i]

    data_tot_avg=np.array(data_tot_avg)

    time_avg = time[::n_avg]

    while len(time_avg)>data_tot_avg.shape[-1]:
        time_avg=time_avg[:-1]

    ch_1,ch_2,ch_3=V_to_strain(data_tot_avg[0],G=1.86),V_to_strain(data_tot_avg[1]),V_to_strain(data_tot_avg[2],G=1.86)
    xx,yy,xy=rosette_to_tensor(ch_1,ch_2,ch_3)

    results=np.array([time_avg,xx,yy,xy])
    np.save(location[:-4]+'_epsilon_time_xx_yy_xy.npy',results)
    return(None)



def transformation2(location,n_avg=1,chan=3):
    time,data_tot= open_data_bin(location)
    fs = round(1/np.median(time[1:]-time[:-1]))
    data_tot_filtered=np.copy(data_tot)
    data_tot_avg=[]
    for i in range(len(data_tot_filtered)):
        data_tot_avg.append(avg_bin(data_tot_filtered[i],n_avg))
        data_tot_avg[i]=data_tot_avg[i]
    data_tot_avg=np.array(data_tot_avg)
    time_avg = time[::n_avg]
    while len(time_avg)>data_tot_avg.shape[-1]:
        time_avg=time_avg[:-1]
    results=np.array([time_avg,data_tot_avg[chan-1],data_tot_avg[3]])
    np.save(location[:-4]+'_time_u{}1_u{}2'.format(chan,chan),results)
    return(None)

#def transformation3(location):



for i in files:
    transformation1(location+i,n_avg=n_avg)
for i in files:
    transformation2(location+i,n_avg=n_avg,chan=chan)

