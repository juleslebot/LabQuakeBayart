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

def find_event(x):
    l=len(x)
    avg=np.mean(x[:l//2-l//10])
    std=np.std(x[:l//2-l//10])
    searchzone=x[l//2-l//10:l//2+l//10]
    i=list(filter(lambda i: (searchzone[i] > avg+3.5*std) or (searchzone[i] < avg-3.5*std), range(l//100+1,len(searchzone)-1-l//100)))[0]
    return(i+l//2-l//10)


def find_event(x):
    l=len(x)
    avg=np.mean(x[:l//2-l//10])
    std=np.std(x[:l//2-l//10])
    searchzone=x[l//2-l//10:l//2+l//10]
    i=list(filter(lambda i: (searchzone[i] > avg+5*std) or (searchzone[i] < avg-5*std), range(l//100+1,len(searchzone)-1-l//100)))[0]
    avg2=np.mean(searchzone[:i-i//10])
    std2=np.std(searchzone[:i-i//10])
    j=list(filter(lambda j: (searchzone[i-l//1000:i+l//1000][j] > avg2+3.7*std2) or (searchzone[i-l//1000:i+l//1000][j] < avg2-3.7*std2), range(2*l//1000-2)))[0]
    return(i+j+l//2-l//10-l//1000)



## 16 channels dictionnary Plot
loc_folder="D:/Users/Manips/Documents/DATA/FRICS/2023/2023-03-13/Full_Daq_4_data/"
#loc_folder="D:/Users/Manips/Documents/DATA/FRICS/2023/2023-03-13/testbruit/"
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




#export_to_matlab={"data":newdata,"ylabels":newylabels,"time":time,"Fn":data[32]}

#scio.savemat(loc_folder+loc_file+".mat",export_to_matlab)




### Find events


fig, axs = plt.subplots(3,5,sharex=True)

for i in range(len(newdata)):
    axs[i%3][i//3].plot(time[start::speedup],newdata[i][start::speedup], label=newylabels[i])
    axs[i%3][i//3].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axs[i%3][i//3].grid("both")
    axs[i%3][i//3].legend()
    print(i)
    #axs[i%3][i//3].axvline(time[find_event(newdata[i])])

axs[-1][0].set_xlabel('time (s)')

fig.set_size_inches(14,8)
plt.show()


###

times=[find_event(x) for x in newdata]
x=[0,2.5,4.5,6.5,8.5]
times[11]=start+19795*speedup

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
    axes[i].axvline(time[int(y[i])])
    axes[i].axvline(time[int(y[i]-y_err[i])],linestyle='dashed')
    axes[i].axvline(time[int(y[i]+y_err[i])],linestyle='dashed')
    axes[i].grid(which="Both")
plt.show()






###



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

def find_event(x):
    l=len(x)
    avg=np.mean(x[:l//2-l//10])
    std=np.std(x[:l//2-l//10])
    searchzone=x[l//2-l//10:l//2+l//10]
    i=list(filter(lambda i: (searchzone[i] > avg+3.5*std) or (searchzone[i] < avg-3.5*std), range(l//100+1,len(searchzone)-1-l//100)))[0]
    return(i+l//2-l//10)


def find_event(x):
    l=len(x)
    avg=np.mean(x[:l//2-l//10])
    std=np.std(x[:l//2-l//10])
    searchzone=x[l//2-l//10:l//2+l//10]
    i=list(filter(lambda i: (searchzone[i] > avg+5*std) or (searchzone[i] < avg-5*std), range(l//100+1,len(searchzone)-1-l//100)))[0]
    avg2=np.mean(searchzone[:i-i//10])
    std2=np.std(searchzone[:i-i//10])
    j=list(filter(lambda j: (searchzone[i-l//100:i+l//100][j] > avg2+3.5*std2) or (searchzone[i-l//100:i+l//100][j] < avg2-3.5*std2), range(2*l//100-2)))[0]
    return(i+j+l//2-l//10-l//100)



## 16 channels dictionnary Plot
loc_folder="D:/Users/Manips/Documents/DATA/FRICS/2023/2023-03-09/"

loc_file="slowmon.npy"
loc=loc_folder+loc_file
sampling_freq_in = 100

speedup=1
speedup_smooth=1
roll_smooth=1
start=1
navg=1

data=np.array(np.load(loc))


fig,axes=plt.subplots(4,1,sharex=True,sharey=False,gridspec_kw={'hspace': 0})
axes[0].plot(data[0])
axes[1].plot(data[15])
axes[2].plot(data[31])
axes[3].plot(data[47])
axes[0].grid(which="Both")
axes[1].grid(which="Both")
axes[2].grid(which="Both")
axes[3].grid(which="Both")
plt.show()

















