## Imports
# Science
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from scipy import interpolate
# DAQ
import nidaqmx
from nidaqmx.stream_readers import AnalogMultiChannelReader
from nidaqmx import constants
# random
import threading
import pickle
import os

try :
    from Python_DAQ import *
except :
    from DAQ_Python.Python_DAQ import *

def convert_123_xy(data_dict):
    data=data_dict["data"]
    ylabels=data_dict["ylabels"]
    for i in range(5):
        data[3*i],data[3*i+1],data[3*i+2]=rosette_to_tensor(data[3*i],data[3*i+1],data[3*i+2])
        ylabels[3*i],ylabels[3*i+1],ylabels[3*i+2]=r"$\epsilon_{{xx}}^{}$".format(i+1),r"$\epsilon_{{yy}}^{}$".format(i+1),r"$\epsilon_{{xy}}^{}$".format(i+1)
    data_dict["data"]=data
    data_dict["ylabels"]=ylabels
    return(data_dict)

## Parameters

ifile           = 7
loc_folder      = "D:/Users/Manips/Documents/DATA/FRICS/2023/2023-03-13/"
loc_file        = "Full_Daq_{}.npy".format(ifile)
speedup         = 1
speedup_smooth  = 10
roll_smooth     = 1
start           = 1
start_of_search = 1000
end_of_search   = 29900
sensit          = 2.5



## Load data
loc=loc_folder+loc_file
#loc="D:/Users/Manips/Downloads/test.npy"

data_dict=np.load(loc,allow_pickle=True).all()
data_dict=convert_123_xy(data_dict)

ylabels=data_dict["ylabels"]
data=data_dict["data"]
data=smooth(data,roll_smooth)
data=np.array([avg_bin(data[i],speedup_smooth) for i in range(len(data))])
data=np.transpose(np.transpose(data)-data[:,start])

sampling_freq_in=data_dict["sampling_freq_in"]/speedup_smooth
navg=data_dict["navg"]

time=np.arange(len(data[0]))/sampling_freq_in*navg


### Create directories

filename=loc_file[:loc_file.find('.')]

locfigures=loc_folder+filename+"_figures/"

folderslist=["","mu/","xx/","xy/","yy/","progress/","Daq/"]

for folder in folderslist :
    try:
        os.mkdir(locfigures+folder)
    except:
        print("folders do already exist")


###


index_list=start_of_search+jump_detect(data[-1][start_of_search:],typical_wait_time=10,sensit=sensit)
if end_of_search!=-1:
    index_list=index_list[index_list<end_of_search]

jump_plot(data[-1],index_list)


jumps=[]
for i in index_list:
    try :
        jumps.append(data[:,i+5]-data[:,i-5])
    except:
        index_list=index_list[:-1]

jumps=np.array(jumps)


data_save={}
data_save["events_index_list"]=index_list*150
data_save["data_before"]=data[:,index_list-2]
data_save["data_after"]=data[:,index_list+2]


np.save(locfigures+"events_index_list_{}.npy".format(ifile), data_save)
scio.savemat(locfigures+"events_index_list_{}.mat".format(ifile),data_save)
###

time_ter=time[index_list-10]
data_ter=data[:,index_list-10]



###
fig, axs = plt.subplots(3,6,sharex=True)

for i in range(len(data)-2):
    axs[i%3][i//3].plot(time[start::speedup],data[i][start::speedup], label=ylabels[i])
    axs[i%3][i//3].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axs[i%3][i//3].grid("both")
    axs[i%3][i//3].legend()
    axs[i%3][i//3].plot(time_ter,data_ter[i])

#for i in range(15):
#    axs[0][0].get_shared_y_axes().join(axs[i%3][0], axs[i%3][i//3])


i=len(data)-2
j=i+1
axs[j%3][j//3].plot(time[start::speedup],data[i][start::speedup], label=ylabels[i])
axs[j%3][j//3].grid("both")
axs[j%3][j//3].legend()
axs[j%3][j//3].plot(time_ter,data_ter[i])

i=len(data)-1
j=i+1
axs[j%3][j//3].plot(time[start::speedup],data[i][start::speedup], label=ylabels[i])
axs[j%3][j//3].grid("both")
axs[j%3][j//3].legend()
axs[j%3][j//3].plot(time_ter,data_ter[i])


axs[-1][0].set_xlabel('time (s)')

fig.set_size_inches(14,8)
plt.suptitle(loc_file)
plt.savefig(locfigures+"Daq/"+"Full_Daq"+".pdf")
plt.savefig(locfigures+"Daq/"+"Full_Daq"+".png")
plt.close()



###


fig, axs = plt.subplots(3,6,sharex=False)

for i in range(len(data)-1):
    axs[i%3][i//3].hist(jumps[:,i])
    axs[i%3][i//3].set_xlabel(ylabels[i])
    axs[i%3][i//3].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    axs[i%3][i//3].grid("both")

i=len(data)-1
j=i+1
axs[j%3][j//3].hist(jumps[:,i])
axs[j%3][j//3].grid("both")
axs[j%3][j//3].set_xlabel(ylabels[i])




fig.set_size_inches(14,8)
plt.tight_layout()
plt.savefig(locfigures+"Daq/"+"Stats.pdf")
plt.savefig(locfigures+"Daq/"+"Stats.png")
plt.close()


###

profile_yy = data[[1,4,7,10,13]]
profile_xy = data[[2,5,8,11,14]]
profile_xx = data[[0,3,6,9,12]]
E=4e9
nu=0.33
S=0.15*0.01
profile_xy_over_yy = - ( (1-nu) / 2 ) * (profile_xy / ( profile_yy + nu * profile_xx ) )

x=np.array([3,5.5,7.5,9.5,11.5])
ti=time[index_list]
ti_before=time[index_list]-0.1
ti_after=time[index_list]+0.1


###
def strain_profile(data,x,time,ti,col="r",y_label="no_label",data_label=".",no_legend=False):
    ti_index=[np.argmax(time>ti_i) for ti_i in ti]
    #
    label=r"t={0[0]:.1f}, $<" + data_label + ">=${0[1]:.1e}"
    if len(ti_index)==1:
        legend_tab=(ti[0], np.mean(data[:,ti_index[0]]))
        y=data[:,ti_index[0]]
        if col=="r":
            color=(1,0,0)
        else :
            color=(0,1,1)
        plt.plot(x, y, color=color, label=label.format(legend_tab),alpha=1)
    else :
        for i in range(len(ti_index)):
            legend_tab=(ti[i], np.mean(data[:,ti_index[i]]))
            y=data[:,ti_index[i]]
            if col=="r":
                color=((i)/(len(ti_index)-1),0,0)
            else :
                color=(0,1-(i)/(len(ti_index)-1),(i)/(len(ti_index)-1))
            plt.plot(x, y, color=color, label=label.format(legend_tab),alpha=0.5)
    plt.xlabel("Position $x$ (cm)")
    plt.ylabel(y_label)
    if not no_legend:
        plt.legend(prop={'size': 6})
    plt.tight_layout()
    plt.suptitle(loc_file+" avant-après.")#loc[31:])
    return(None)

###
"""
for i in range(2):
    strain_profile(profile_yy,x,time,[ti_before[i]],col="r")
    strain_profile(profile_yy,x,time,[ti_after[i]],col="b")
    plt.grid(which="both")
    plt.show()
"""







###

i=len(ti)-1

strain_profile(profile_xy_over_yy,x,time,ti_before,col="r",y_label=r"$\mu$",no_legend=True)
ax = plt.gca()
mulim=(-2,3)
ax.set_ylim(mulim)
plt.grid(which="both")
plt.savefig(locfigures+"progress/"+"mu_progress_before.png")
plt.close()

strain_profile(1000*profile_xx,x,time,ti_before,col="r",y_label=r"$\varepsilon_{xx}$ (mStrain)",no_legend=True)
plt.grid(which="both")
plt.savefig(locfigures+"progress/"+"xx_progress_before.png")
ax = plt.gca()
xxlim=ax.get_ylim()
plt.close()

strain_profile(1000*profile_xy,x,time,ti_before,col="r",y_label=r"$\varepsilon_{xy}$ (mStrain)",no_legend=True)
plt.grid(which="both")
plt.savefig(locfigures+"progress/"+"xy_progress_before.png")
ax = plt.gca()
xylim=ax.get_ylim()
plt.close()

strain_profile(1000*profile_yy,x,time,ti_before,col="r",y_label=r"$\varepsilon_{yy}$ (mStrain)",no_legend=True)
ax = plt.gca()
yylim=[-1,2]
ax.set_ylim(yylim)
#yylim=ax.get_ylim()
plt.grid(which="both")
plt.savefig(locfigures+"progress/"+"yy_progress_before.png")
plt.close()

strain_profile(profile_xy_over_yy,x,time,ti_after,col="r",y_label=r"$\mu$",no_legend=True)
ax = plt.gca()
ax.set_ylim(mulim)
plt.grid(which="both")
plt.savefig(locfigures+"progress/"+"mu_progress_after.png")
plt.close()

strain_profile(1000*profile_xx,x,time,ti_after,col="r",y_label=r"$\varepsilon_{xx}$ (mStrain)",no_legend=True)
plt.grid(which="both")
plt.savefig(locfigures+"progress/"+"xx_progress_after.png")
plt.close()

strain_profile(1000*profile_xy,x,time,ti_after,col="r",y_label=r"$\varepsilon_{xy}$ (mStrain)",no_legend=True)
plt.grid(which="both")
plt.savefig(locfigures+"progress/"+"xy_progress_after.png")
plt.close()

strain_profile(1000*profile_yy,x,time,ti_after,col="r",y_label=r"$\varepsilon_{yy}$ (mStrain)",no_legend=True)
plt.grid(which="both")
plt.savefig(locfigures+"progress/"+"yy_progress_after.png")
plt.close()




###

#Fn_moy=np.mean(...)
#eps_yy_moy=Fn_moy/(S*E)

###


for i in range(len(ti_before)):
    strain_profile(1000*profile_xx,x,time,[ti_before[i]],col="r",y_label=r"$\varepsilon_{xx}$ (mStrain)")
    strain_profile(1000*profile_xx,x,time,[ti_after[i]],col="b",y_label=r"$\varepsilon_{xx}$ (mStrain)")
    ax = plt.gca()
    ax.set_ylim(xxlim)
    plt.grid(which="both")
    plt.tight_layout()
    plt.savefig(locfigures+"xx/"+"xx{}.png".format(i))
    plt.close()

for i in range(len(ti_before)):
    strain_profile(1000*profile_yy,x,time,[ti_before[i]],col="r",y_label=r"$\varepsilon_{yy}$ (mStrain)")
    strain_profile(1000*profile_yy,x,time,[ti_after[i]],col="b",y_label=r"$\varepsilon_{yy}$ (mStrain)")
    ax = plt.gca()
    ax.set_ylim(yylim)
    #ax.axhline(eps_yy_moy,linestyle="--",color="grey",alpha=0.5)
    plt.grid(which="both")
    plt.tight_layout()
    plt.savefig(locfigures+"yy/"+"yy{}.png".format(i))
    plt.close()

for i in range(len(ti_before)):
    strain_profile(1000*profile_xy,x,time,[ti_before[i]],col="r",y_label=r"$\varepsilon_{xy}$ (mStrain)")
    strain_profile(1000*profile_xy,x,time,[ti_after[i]],col="b",y_label=r"$\varepsilon_{xy}$ (mStrain)")
    ax = plt.gca()
    ax.set_ylim(xylim)
    plt.grid(which="both")
    plt.tight_layout()
    plt.savefig(locfigures+"xy/"+"xy{}.png".format(i))
    plt.close()

for i in range(len(ti_before)):
    strain_profile(profile_xy_over_yy,x,time,[ti_before[i]],col="r",y_label=r"$\mu$")
    strain_profile(profile_xy_over_yy,x,time,[ti_after[i]],col="b",y_label=r"$\mu$")
    ax = plt.gca()
    ax.set_ylim(mulim)
    plt.grid(which="both")
    plt.tight_layout()
    plt.savefig(locfigures+"mu/"+"mu{}.png".format(i))
    plt.close()




###

fig,ax=plt.subplots(2,1,sharex=True)

for tis in ti:
    ax[0].axvline(tis,linestyle="--",color="grey",alpha=0.5)
    ax[1].axvline(tis,linestyle="--",color="grey",alpha=0.5)

ax[0].plot(time,data[-2])
ax[1].plot(time,data[-1])
ax[0].grid(which="both")
ax[1].grid(which="both")
ax[1].set_xlabel("Time (s)")
ax[0].set_ylabel(ylabels[-2])
ax[1].set_ylabel(ylabels[-1])



plt.suptitle(loc_file + r"   avg $\sigma_N =$ {:.3e} Pa".format(np.mean(data[-1][1000:]) / S ))
plt.savefig(locfigures+"Daq/"+"forces.pdf")
plt.savefig(locfigures+"Daq/"+"forces.png")
plt.close()

### Save parameters

variables_to_save={\
    "loc_folder":loc_folder,\
    "loc_file":loc_file,\
    "speedup":speedup,\
    "speedup_smooth":speedup_smooth,\
    "roll_smooth":roll_smooth,\
    "start":start,\
    "start_of_search":start_of_search,\
    "end_of_search":end_of_search,\
    "index_list":index_list,\
    "sensit":sensit,\
    "E":E,\
    "nu":nu,\
    "S":S\
    }

np.save(locfigures+"Python_parameters.npy",variables_to_save)




