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


## Load
data=np.loadtxt("D:/Users/Manips/Documents/DATA/FRICS/2021-02-08/DIC1/DAQ.txt")

## Full Plot
f=quick_plot_data(data,sampling_freq_in,ylabels=ylabels)
plt.show()

## Superposition Plot
def superposed_plot(data,sampling_freq_in):
    f, ax = plt.subplots(2, 1, sharex='all', sharey='none')
    x=np.arange(data[0,:].size)/sampling_freq_in
    ax[1].plot(x,data[2,:],label = "left")
    ax[1].plot(x,data[3,:],label="right")
    ax[1].set_ylabel("d (mm)")
    ax[1].set_xlabel("t (s)")
    ax[1].grid(True)

    ax[0].plot(x,data[0,:])
    ax[0].set_ylabel("Fn (kg)")
    ax[0].grid(True)

    plt.legend()
    plt.show()
    return(None)


superposed_plot(data,sampling_freq_in)

## Force-delta plot
def force_delta(data,show=True):
    plt.scatter(data[0,:],data[2,:]-data[3,:],s=1)
    plt.ylabel(r"$\Delta$ d (mm)")
    plt.xlabel("F (kg)")
    plt.grid(True)
    if show:
        plt.show()
    return(None)

##
data=np.loadtxt("D:/Users/Manips/Documents/DATA/FRICS/2020-02-04/test_rotation.txt")
force_delta(data)



## A new plot (05/02)
col=[ "darkgreen","mediumseagreen","lime","navy","blue","deepskyblue" ]
lab=["control","loose jaw","wedge","control bis","loose jaw bis","wedge bis"]

n_avg=100
f,ax=plt.subplots(2,3,sharex="all",sharey="all")
for i in range(6):
    for j in range(5):
        data=np.loadtxt("D:/Users/Manips/Documents/DATA/FRICS/2020-02-05/{}-{}.txt".format(i+1,j+1))
        deltad=data[2,:]-data[3,:]
        #deltad = deltad-np.average(deltad) # Wait, that's illegal !
        deltad=moving_average(deltad,n_avg)
        force=moving_average(data[0,:],n_avg)
        ax[i//3][i%3].scatter(force,deltad,s=.5,color=col[i])
    ax[i//3][i%3].set_title(lab[i])
    ax[i//3][i%3].grid(True)
    ax[i//3][i%3].tick_params(labelbottom=True)
    ax[i//3][i%3].tick_params(labelleft=True)
ax[-1,0].set_ylabel(r"$\Delta$ d (mm)")
ax[-1,0].set_xlabel("F (kg)")
plt.show()

## An other type

col=[ "darkgreen","mediumseagreen","lime","navy","blue","deepskyblue" ]
lab=["control","loose jaw","wedge","control bis","loose jaw bis","wedge bis"]

n_avg=100
f,ax=plt.subplots(1,3,sharex="all",sharey="all")
for i in range(6):
    for j in range(5):
        data=np.loadtxt("D:/Users/Manips/Documents/DATA/FRICS/2020-02-05/{}-{}.txt".format(i+1,j+1))
        deltad=data[2,:]-data[3,:]
        #deltad = deltad-np.average(deltad) # Wait, that's illegal !
        deltad=moving_average(deltad,n_avg)
        force=moving_average(data[0,:],n_avg)
        if j==0:
            ax[i%3].scatter(force,deltad,s=.5,color=col[i],alpha=.5,label=lab[i])
        else:
            ax[i%3].scatter(force,deltad,s=.5,color=col[i],alpha=.5)
    ax[i%3].legend(markerscale=10.)
    ax[i%3].set_title(lab[i%3])
    ax[i%3].grid(True)
    ax[i%3].tick_params(labelbottom=True)
    ax[i%3].tick_params(labelleft=True)
ax[0].set_ylabel(r"$\Delta$ d (mm)")
ax[0].set_xlabel("F (kg)")
plt.show()

## Plot Fs(Fn) :

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


a=np.loadtxt("D:/Users/Manips/Documents/DATA/FRICS/2021-02-15/results.txt")
xmax=1.1*np.max(a[:,1])
ymax=1.1*np.max(a[:,2])
xrange=np.array([0,xmax])

def split_speed(a):
    b=a[a[:,0].argsort()]
    ind=np.where(b[:-1,0] != b[1:,0])[0]+1
    c=np.split(b,ind)
    return(c)

c=split_speed(a)

ax=plt.gca()

for i in c:
    color = next(ax._get_lines.prop_cycler)['color']
    #plt.errorbar(i[:,1],np.abs(i[:,2]),xerr=2,yerr=2,linestyle=None,fmt="+",label="{} mm/s".format(i[0,0]),alpha=0.7)
    X=i[:,1].reshape((-1,1))
    y=np.abs(i[:,2])
    reg=LinearRegression().fit(X,y)
    r=reg.score(X, y)
    alpha=reg.coef_[0]
    plt.plot(xrange,reg.predict(xrange.reshape((-1,1))),color=color)
    plt.scatter(i[:,1],y,label="$v=${} mm/s, $r^2\simeq${:.3f}, $\\alpha\simeq${:.2f}".format(i[0,0],r**2,alpha),marker="+",alpha=0.7,color=color)


plt.legend()
plt.xlabel('$F_n$ (kg)')
plt.ylabel('$F_s$ (kg)')
plt.xlim(0,xmax)
plt.ylim(0,ymax)
plt.grid(which="both")


plt.show()

