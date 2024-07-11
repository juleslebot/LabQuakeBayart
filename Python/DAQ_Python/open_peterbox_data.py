"""
This files is used to pick for each event of each experiment the moment at which
the crack is detected on each gage.
"""

## Imports
# Science
import matplotlib.pyplot as plt
import numpy as np

# custom file
try :
    from Python_DAQ import *
except :
    from DAQ_Python.Python_DAQ import *



# Location
# main folder and file
loc_folder="E:/2023-2024/2024-06-07_test-nv-bloc/"
loc_file="event-002.npy"
# name of the reference file, containing unloaded signal, usually event-001
# name of the parameters file
loc_file_zero = "event-001.npy"
loc_params="parameters.txt"

loc=loc_folder+loc_file
loc_zero=loc_folder+loc_file_zero


# control smoothing of the data (rolling average) and starting point
roll_smooth=30
start=0

## gages of interest and other channels
# if you want them all :
gages_true_number = np.arange(1,61,1)

# if you want just a specific set of gages
# gages_true_number = [1,2,3,7,8,9,58,59,60]
# if you want all center gages
# gages_true_number = np.arange(2,61,3)



# convert gage number into channel number
gages_channels = gauge_number_to_channel(gages_true_number)
nchannels = len(gages_channels)

# channels containing the normal and shear force, and the trigger
forces_channels = [15,31]
trigger_channel = 47




## Load Parameters
# if x is not set in the parameters, set a default x :
x=np.linspace(0.006,0.138,20)
exec(load_params(loc_folder+loc_params))
sampling_freq_in = clock/10


## Fast acquicition
# Load data
data=np.load(loc,allow_pickle=True)
data_zero=np.load(loc_zero,allow_pickle=True)

# smooth data
data=smooth(data,roll_smooth)
data=np.transpose(np.transpose(data)-np.mean(data_zero,axis=1))

# assign specific channels to specific variables
forces=data[forces_channels,:]
mu = data[forces_channels[1],:].mean() / data[forces_channels[0],:].mean()
gages = data[gages_channels]
gages_zero = data_zero[gages_channels]
fast_time=np.arange(len(gages[0]))/sampling_freq_in


## load slow monitoring
sm = np.load(loc_folder+"slowmon.npy",allow_pickle=True)
time_sm = np.load(loc_folder+"slowmon_time.npy")

# extract observables
gages_sm = sm[gages_channels]
gages_sm=np.transpose(np.transpose(gages_sm)-np.mean(gages_zero,axis=1))


forces_sm=sm[forces_channels,:]
fn_sm=forces_sm[0]
fs_sm=forces_sm[1]

trigger = 100*sm[trigger_channel]

converted = False

### Convert voltage to strains or forces, and rosette to tensor

if not converted:
    for i in range(nchannels//3):
        ch_1=gages[3*i]
        ch_2=gages[3*i+1]
        ch_3=gages[3*i+2]
        ch_1,ch_2,ch_3=voltage_to_strains(ch_1,ch_2,ch_3,amp=2000)
        ch_1,ch_2,ch_3=rosette_to_tensor(ch_1,ch_2,ch_3)
        gages[3*i]=ch_1
        gages[3*i+1]=ch_2
        gages[3*i+2]=ch_3

    for i in range(nchannels//3):
        ch_1=gages_zero[3*i]
        ch_2=gages_zero[3*i+1]
        ch_3=gages_zero[3*i+2]
        ch_1,ch_2,ch_3=voltage_to_strains(ch_1,ch_2,ch_3,amp=2000)
        ch_1,ch_2,ch_3=rosette_to_tensor(ch_1,ch_2,ch_3)
        gages_zero[3*i]=ch_1
        gages_zero[3*i+1]=ch_2
        gages_zero[3*i+2]=ch_3

    for i in range(nchannels//3):
        ch_1=gages_sm[3*i]
        ch_2=gages_sm[3*i+1]
        ch_3=gages_sm[3*i+2]
        ch_1,ch_2,ch_3=voltage_to_strains(ch_1,ch_2,ch_3,amp=2000)
        ch_1,ch_2,ch_3=rosette_to_tensor(ch_1,ch_2,ch_3)
        gages_sm[3*i]=ch_1
        gages_sm[3*i+1]=ch_2
        gages_sm[3*i+2]=ch_3


    eps_yy_sm = gages_sm[1::3]
    eps_xy_sm = gages_sm[2::3]
    eps_xx_sm = gages_sm[0::3]


    forces = voltage_to_force(forces)
    forces_sm=voltage_to_force(forces_sm)
    fn_sm = voltage_to_force(fn_sm)
    fs_sm = voltage_to_force(fs_sm)

converted=True



### Plot macroscopic checks

#
# # slowmon forces and trigger
# plt.figure()
# plt.title("Fn, Fs and trigger")
# plt.suptitle(loc_folder+loc_file,alpha=.2,size=5)
#
#
# #plt.plot(time_sm,fn_sm)
# time_ac = np.linspace(0,len(ch1)/1e+6,len(ch1))
# f_ech_sm = len(time_sm)/(time_sm[-1]-time_sm[0])
# f_ech_ac = 1e+6
# A = 3000*f_ech_sm/f_ech_ac
# plt.plot(time_sm-60.15,fs_sm)
# plt.plot(time_sm-60.15,trigger)
# plt.plot(time_ac,ch1/10)
#
#
# plt.xlabel("time (s)")
# plt.grid()
#
# #plt.savefig(loc_folder+"slowmon_forces.png")
#
# plt.show()
#
#


# strain profile

plt.figure()
plt.title("Strain profile (m/m)")
plt.suptitle(loc_folder+loc_file,alpha=.2,size=5)



plt.scatter(x,np.mean(gages[1::3,:1000],axis=1))
plt.xlabel("position (m)")
plt.ylabel(r"$\varepsilon_{yy}$ (m/m)")
plt.grid(which="both")
plt.ylim((1.1*min(0,min(np.mean(gages[1::3,:1000],axis=1))),1.1*max(np.mean(gages[1::3,:1000],axis=1))))
plt.savefig(loc_folder+"{}_load_profile.png".format(loc_file[:-4]))

plt.show()

###

#



