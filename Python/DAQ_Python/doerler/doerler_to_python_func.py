## Imports
# Science
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy import signal
import os

from scipy import interpolate
try :
    from Python_DAQ import *
except :
    from DAQ_Python.Python_DAQ import *



## read csv

location='D:/Users/Manips/Documents/DATA/FRICS/2022/2022-06-07/Exports/'
#file = 'ens_2022_06_07_081311_1.csv'

delimiter=','
usecols=np.arange(1,7)


list_files=[]
for file in os.listdir(location):
    if file.endswith(".csv"):
        list_files.append(file)


def convert_from_doerler(location,file):

    with open(location+file) as f:
        names=f.readline()
        names=names[:-1]
        names=names.split(sep=delimiter)
        time_index=names.index('Time (s)')
        names=list(np.delete(names,time_index))
        loaded_data=f.readlines()
        for i in range(len(loaded_data)):
            loaded_data[i]=loaded_data[i].split(sep=delimiter)
            try:
                index=loaded_data[i][time_index].index(':')
                loaded_data[i][time_index]=str(int(loaded_data[i][time_index][:index])*60+float(loaded_data[i][time_index][index+1:]))
            except:
                None
            loaded_data[i][-1]=loaded_data[i][-1][:-1]

    loaded_data=np.array(loaded_data,dtype=float)
    loaded_data=loaded_data.transpose()


    time=loaded_data[time_index]
    data_tot=np.delete(loaded_data,time_index,axis=0)
    data_tot=data_tot/1000000

    data_tot_filtered=np.copy(data_tot)


    i=0
    invert=1
    ch_1,ch_2,ch_3=invert*data_tot[i],invert*data_tot[i+1],invert*data_tot[i+2]
    xx,yy,xy=rosette_to_tensor(ch_1,ch_2,ch_3)

    results=np.array([time,xx,yy,xy])
    np.save(location+file[:-4]+'_R07_epsilon_time_xx_yy_xy.npy',results)

    i=3
    invert=1
    ch_1,ch_2,ch_3=invert*data_tot[i],invert*data_tot[i+1],invert*data_tot[i+2]
    xx,yy,xy=rosette_to_tensor(ch_1,ch_2,ch_3)

    results=np.array([time,xx,yy,xy])
    np.save(location+file[:-4]+'_R10_epsilon_time_xx_yy_xy.npy',results)


    #other data
    ch_1,ch_2,ch_3,ch_4=data_tot[6],data_tot[1],data_tot[4],data_tot[7]

    results=np.array([time,ch_1,ch_2,ch_3,ch_4])
    np.save(location+file[:-4]+'_J2_R04_07_10_12_epsilon_time.npy',results)


for file in list_files:
    convert_from_doerler(location,file)


