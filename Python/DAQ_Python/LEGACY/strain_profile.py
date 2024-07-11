## Imports
# Science
import matplotlib.pyplot as plt
import numpy as np
try :
    from Python_DAQ import *
except :
    from DAQ_Python.Python_DAQ import *


#ylabels=["Fn (kg)", "Fs (kg)", "d left (mm)", "d right (mm)"]
### Constants
sampling_freq_in=100000
navg=100
x=np.array([3,5.5,7.5,9.5,11.5])
ylabels=["$F_n$ (kg)", "$F_f$ (kg)", "TTL", "chan 2","chan 5","chan 8","chan 11","chan 14"]
loc="D:/Users/Manips/Documents/DATA/FRICS/2022/2022-10-06-references/lots_of_grain/100kg.txt"



data=np.loadtxt(loc)
time=np.arange(len(data[0]))/sampling_freq_in*navg


### Interesting times

time_start=time[np.where(np.abs(smooth(data[1],1000))>5)[0][0]]

time_events=time[ np.where(np.diff(data[2]>0))[0][0::1] ]

time_events_clean=[time_events[0]]

for i in range(1,len(time_events)):
    if np.abs((time_events_clean[-1]-time_events[i]))>0.1:
        time_events_clean.append(time_events[i])

time_events_clean=np.array(time_events_clean)



ti=[time_start-3]+list(time_events_clean[::10]-0.1)

### plot

def strain_profile(data,x,time,ti):
    ti_index=[np.argmax(time>ti_i) for ti_i in ti]
    label="t={0[0]:.1f}, $F_n$={0[1]:.1f}, $F_f$={0[2]:.1f}, $<\epsilon>=${0[3]:.1e}"
    for i in range(len(ti_index)):
        legend_tab=(ti[i], data[0,ti_index[i]], data[1,ti_index[i]], np.mean(data[3:,ti_index[i]]))
        y=data[3:,ti_index[i]]
        if i==0:
            color=(0,0.8,0.8)
        else:
            color=((i)/(len(ti_index)-1),0,0)
        plt.plot(x, y, color=color, label=label.format(legend_tab))
    plt.xlabel("Position $x$ (cm)")
    plt.ylabel(r"Déformation $\epsilon$")
    plt.legend(prop={'size': 6})
    plt.grid(which="both")
    plt.tight_layout()
    plt.suptitle(loc[31:])
    plt.show()
    return(None)



strain_profile(data,x,time,ti)