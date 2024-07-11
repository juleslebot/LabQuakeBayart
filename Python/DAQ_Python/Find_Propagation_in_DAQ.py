## Quick and dirty check
## Imports
# Science
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from scipy import interpolate


try :
    from Python_DAQ import *
except :
    from DAQ_Python.Python_DAQ import *



## 16 channels dictionnary Plot
loc_folder="D:/Users/Manips/Documents/DATA/FRICS/2023/2023-03-13/Full_Daq_4_data/"
#loc_folder="C:/FTP_Root/DATA/"

loc_file="event-005.npy"
loc=loc_folder+loc_file
sampling_freq_in = 8000000

speedup=1
speedup_smooth=1
roll_smooth=100
start=1
navg=1

data=np.load(loc)

ylabels=["CH{}".format(i) for i in range(1,16)]

data=smooth(data,roll_smooth)
data=np.array([avg_bin(data[i],speedup_smooth) for i in range(len(data))])

data=np.transpose(np.transpose(data)-data[:,start])

sampling_freq_in=sampling_freq_in/speedup_smooth

time=np.arange(len(data[0]))/sampling_freq_in*navg

newdata=[rosette_to_tensor(data[i],data[i+1],data[i+2]) for i in range(5)]
temp=[]
for i in newdata:
    for j in i:
        temp.append(j)

newdata=np.array(temp)
newylabels=[r"$\epsilon_{xx}^",r"$\epsilon_{yy}^",r"$\epsilon_{xy}^"]*5
for i in range(15):
    newylabels[i]=newylabels[i] + str(1+i//3) + "$"




### Find events

# Randomize the order, to ensure no experimentators bias

n=15
randomorder=np.random.permutation(n)

L = [ (randomorder[i],i) for i in range(n) ]
L.sort()
_,permutation = zip(*L)
permutation=np.array(permutation)

###

for i in range(n):
    print(i)
    plt.plot(newdata[randomorder[i]])
    plt.grid(which="both")
    plt.show()

###

indexes=[19681,
19683,
19645,
19757,
19800,
19710,
19750,
19662,
19761,
19672,
19510,
19672,
19762,
19654,
19756]

indexes=np.array(indexes)

###

times=indexes[permutation]

np.savetxt(loc+"_times_hand_picked.txt",list(times))

x=[0,2.5,4.5,6.5,8.5]

y=[np.mean(times[3*i:3*i+3]) for i in range(5)]
y_err=[np.std(times[3*i:3*i+3]) for i in range(5)]
plt.scatter(x,times[::3])
plt.scatter(x,times[1::3])
plt.scatter(x,times[2::3])
plt.errorbar(x,y,yerr=y_err)
plt.show()


###

fig,axes=plt.subplots(5,1,sharex=True,sharey=False,gridspec_kw={'hspace': 0})
for i in range(5):
    axes[i].plot(time,newdata[3*i+2])
    #axes[i].axvline(time[int(y[i])])
    #axes[i].axvline(time[int(y[i]-y_err[i])],linestyle='dashed')
    #axes[i].axvline(time[int(y[i]+y_err[i])],linestyle='dashed')
    axes[i].grid(which="Both")
plt.show()


