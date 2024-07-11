import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import fnmatch
import scipy.io as scio
import glob

try :
    sys.path.insert(0, "D:/Users/Manips/Documents/Python/DAQ_Python")
    from Python_DAQ import *
except :
    from DAQ_Python.Python_DAQ import *




## Data location

loc_folder="D:/Users/Manips/Documents/DATA/FRICS/2023/2023-07-11-manips-10-voies/"
loc_folder="E:/2023-2024/2023-07-11-manips-10-voies/"

#loc_folder="E:/2023-2024/2023-12-01-low_FN_measurement/"

loc_manip = "manip_{}/"
save_loc = "python_plots/"
loc_old_data = "D:/Users/Manips/Documents/DATA/FRICS/2023/2023-01-05-manip-grains/summary.npy"
gages_channels = np.concatenate([np.arange(0,15),np.arange(16,31)])
forces_channels = np.array([32,33])
nchannels = len(gages_channels)
trigger_channel = 34

show=False

## load daq data


to_remove = [7,19,20,21,22,23,24,25,32,41]
#to_remove=[]
n_manip = 44
#n_manip = 4

solids = [14,15,16,17,18,37,38]
#solids = []
weird = [32,33]
#weird = []
manips = ["manip_{}".format(i) for i in range(1,n_manip+1) if i not in to_remove]
manip_num=[int(m[6:]) for m in manips]


# Initialise global data storage
# in the end we want a list with an element for each experiment
# the element for each experiment will be either a list with the value of
# the parameter for each event of the experiment, or a value for the whole exp.

# LC before the event
loading_contrast = []
# force drop associated with the event
force_drop = []
# fn befor the event
fn = []
fn_aft = []
# fs before the event
fs = []
fs_aft = []
# slowmon trigger channel for each xp
sm_trigs = []
# real time for the sowmon of each xp
sm_times = []
# name of each file
tot_files=[]
# eps yy before each event
eps_yy_bef=[]
# eps yy after each event
eps_yy_aft=[]


sm_eps_yy_mean=[]
sm_eps_xx_mean=[]
sm_eps_xy_mean=[]
sm_fn =[]
sm_fs=[]
eps_bef=[]
eps_aft=[]

# Load AAAAAALL the data.
# I mean All. It takes time.

for manip in manips:
    # Initialise the experiment's values
    lc_temp = []
    force_drops = []
    fns = []
    fss = []
    fns_aft = []
    fss_aft = []
    eps_yy_bef_manip=[]
    eps_yy_aft_manip=[]
    eps_bef_manip=[]
    eps_aft_manip=[]
    # add slowmon values needed
    directory_path = loc_folder + manip + "/"
    sm_times.append(np.load(directory_path+"slowmon_time.npy"))
    sm = np.load(directory_path+"slowmon.npy")

    gages  = sm[gages_channels]
    forces = sm[forces_channels]*500/3
    trig   = sm[trigger_channel]

    # substract initial value
    gages=np.transpose(np.transpose(gages)-np.mean(gages[:,0:100],axis=-1))

    for i in range(nchannels//3):
        ch_1=gages[3*i]
        ch_2=gages[3*i+1]
        ch_3=gages[3*i+2]
        ch_1,ch_2,ch_3=voltage_to_strains(ch_1,ch_2,ch_3)
        ch_1,ch_2,ch_3=rosette_to_tensor(ch_1,ch_2,ch_3)
        gages[3*i]=ch_1
        gages[3*i+1]=ch_2
        gages[3*i+2]=ch_3


    # append slowmons
    sm_trigs.append(trig)
    sm_eps_xx_mean.append(np.mean(gages[0::3],axis=0))
    sm_eps_yy_mean.append(np.mean(gages[1::3],axis=0))
    sm_eps_xy_mean.append(np.mean(gages[2::3],axis=0))
    sm_fn.append(forces[0])
    sm_fs.append(forces[1])

    # create path for per-event data
    directory_path = loc_folder + manip + "/"
    pattern = 'event-[0-9][0-9][0-9].npy'
    matching_files = glob.glob(f"{directory_path}/{pattern}")
    _=matching_files.pop(0)
    tot_files.append(matching_files)
    for i in range(len(matching_files)):
        matching_files[i]=matching_files[i][-13:]

    print(manip)
    # per event data
    for file in matching_files :
        print(file)
        daqdata = np.load(directory_path+file[:13])

        # forces
        forces = daqdata[forces_channels]*500/3
        forces[0]=forces[0]*(1 if np.mean(forces[0])>0 else -1)
        forces[1]=forces[1]*(1 if np.mean(forces[1])>0 else -1)
        fns.append(np.mean(forces[0][:10000]))
        fss.append(np.mean(forces[1][:10000]))
        fns_aft.append(np.mean(forces[0][-1000:]))
        fss_aft.append(np.mean(forces[1][-1000:]))
        force_drops.append(np.mean(forces[1][:10000])-np.mean(forces[1][-10000:]))

        # gages
        gages  = daqdata[gages_channels]
        gages_zero = np.load(directory_path+"event-001.npy")[gages_channels]
        # substract initial value
        gages=np.transpose(np.transpose(gages)-np.mean(gages_zero,axis=-1))

        for i in range(nchannels//3):
            ch_1=gages[3*i]
            ch_2=gages[3*i+1]
            ch_3=gages[3*i+2]
            ch_1,ch_2,ch_3=voltage_to_strains(ch_1,ch_2,ch_3)
            ch_1,ch_2,ch_3=rosette_to_tensor(ch_1,ch_2,ch_3)
            gages[3*i]=ch_1
            gages[3*i+1]=ch_2
            gages[3*i+2]=ch_3

        avg_before = np.mean(gages[:,:1000],axis=-1)
        avg_after = np.mean(gages[:,-1000:],axis=-1)
        eps_yy=avg_before[1::3]
        eps_yy_bef_manip.append(avg_before[1::3])
        eps_yy_aft_manip.append(avg_after[1::3])
        eps_bef_manip.append(avg_before)
        eps_aft_manip.append(avg_after)


        """
        med = np.median(eps_yy[[i for i in range(10) if i!=5]])
        med_tot = np.median(eps_yy[[i for i in range(10)]])
        med_pond= (3*eps_yy[5]+12*med)/15

        lc_temp.append( (eps_yy[5] - med) / med )
        """

        moy = np.mean(eps_yy[[i for i in range(10) if i!=5]])
        moy_pond= (3*eps_yy[5]+12*moy)/15

        #lc_temp.append( (eps_yy[5] - moy) / moy_pond )

        # def comp_lc(prof,fn):
        #     S=0.01*0.015
        #     ss = np.mean(prof[[0,1,2,3,4,6,7,8,9]])
        #     gg = prof[5]
        #     lc = (gg-ss)/(10*fn/S)
        #     return(lc)
        # lc_temp.append(comp_lc( eps_yy , fns[-1] ))


        def comp_lc(prof):
            ss = np.mean(prof[[0,1,2,3,4,6,7,8,9]])
            gg = prof[5]
            lc = 15*(gg-ss)/(3*gg+12*ss)
            return(lc)
        lc_temp.append(comp_lc( eps_yy ))



    fn.append(fns)
    fs.append(fss)
    fn_aft.append(fns_aft)
    fs_aft.append(fss_aft)
    force_drop.append(force_drops)
    eps_yy_aft.append(eps_yy_aft_manip)
    eps_yy_bef.append(eps_yy_bef_manip)
    eps_aft.append(eps_aft_manip)
    eps_bef.append(eps_bef_manip)
    loading_contrast.append(lc_temp[-30:])





## load matlab data for creep
matloc = ["Full_Daq_{}_figures".format(i) for i in range(1,n_manip+1) if i not in to_remove]

creep_left=[]
creep_right=[]
creep_center=[]

for mat in matloc:
    matpath = loc_folder + mat + "/creep/creep2_data.mat"
    matdata = scio.loadmat(matpath)
    creep_left.append(matdata["pourcent_left"][0])
    creep_right.append(matdata["pourcent_right"][0])
    creep_center.append(matdata["pourcent_center"][0])


## extract frequency from the trigger
freqs=[]
deltats=[]


for i in range(len(sm_trigs)):
    sm_trig = sm_trigs[i]
    sm_trig=sm_trig>3
    sm_trig[4:-1]+=sm_trig[0:-5]
    sm_trig=sm_trig>0.5
    stop = len(sm_trig)
    j=10000
    timings=[]
    while j < stop:
        if sm_trig[j] :
            k=0
            while j+k<stop and sm_trig[j+k]:
                k+=1
            if k>25:
                timings.append(j)
            j=j+k
        else :
            j+=1
    if i ==22:
        timings=[timings[j] for j in range(len(timings)) if j!=6 ]
    #on normalise par fn/300
    timings = sm_times[i][timings]#/(np.mean(fn[i])/300)
    deltat = np.diff(timings)
    freq = 1/deltat
    freqs.append(freq)
    deltats.append(deltat)



mean_freq = np.array([np.mean(freq) for freq in freqs])
wide_freq = np.array([[np.quantile(freq,.25),np.quantile(freq,.75)] for freq in freqs]).T
wide_freq=np.abs(wide_freq-mean_freq)
sigma_freq = np.array([np.std(freq) for freq in freqs])
mean_dt = np.array([np.mean(dt) for dt in deltats])
sigma_dt = np.array([np.std(dt) for dt in deltats])

## extract LC from the gages
mean_lc=np.array([np.mean(lc) for lc in loading_contrast])
sigma_lc=np.array([np.std(lc) for lc in loading_contrast])
wide_lc = np.array([[np.quantile(lc,.25),np.quantile(lc,.75)] for lc in loading_contrast]).T
wide_lc=np.abs(wide_lc-mean_lc)


std_eps_yy_ss=E*np.array([np.std(np.array(eps_yy_bef[i])[:,[0,1,2,3,4,6,7,8,9]]) for i in range(len(eps_yy_bef))])
mean_eps_yy_ss=E*np.array([np.mean(np.array(eps_yy_bef[i])[:,[0,1,2,3,4,6,7,8,9]]) for i in range(len(eps_yy_bef))])

mean_fd=np.array([np.mean(fd) for fd in force_drop])
sigma_fd=np.array([np.std(fd) for fd in force_drop])

## extract inter event slip
tot_creep_left = np.array([np.median(c[-5:]) for c in creep_left])
tot_creep_right = np.array([np.median(c[-5:]) for c in creep_right])
tot_creep_center = np.array([np.median(c[-5:]) for c in creep_center])
sigma_creep_left = np.array( [ np.std( c[-5:] ) for c in creep_left])
sigma_creep_right = np.array( [ np.std( c[-5:] ) for c in creep_right])
sigma_creep_center = np.array( [ np.std( c[-5:] ) for c in creep_center])

creep_tot = tot_creep_center-(tot_creep_left+tot_creep_right)/2
creep_sides = (tot_creep_left+tot_creep_right)/2
creep_center = tot_creep_center

## Old data

old_data = np.load(loc_old_data,allow_pickle=True).all()

lc_old=old_data["lc"]
lc_err_old=old_data["lc_err"]
freq_old=old_data["freq"]
freq_err_old=old_data["freq_err"]
ie_slip_old=old_data["ie_slip"]
ie_slip_err_old=old_data["ie_slip_err"]




## define the plot tool
def nice_plot(xpos,ypos,xerr=None,yerr=None,
            xlabel="",ylabel="",annotate=True,save=False,savemat=False,show=True,old_data=False):
    fig, ax = plt.subplots()
    if old_data:
        plt.errorbar(xpos,ypos, xerr=xerr,yerr=yerr,
                    fmt="o",capsize=3,color="green",label="New data")
        plt.errorbar(old_data[0],old_data[1],xerr=old_data[2],yerr=old_data[3],
                    fmt=' ',capsize=3, marker='o',alpha=1,label="old data",color="orange")
        plt.legend()
    else:
        plt.errorbar(xpos,ypos, xerr=xerr,yerr=yerr,
                    fmt="o",capsize=3,color="green")

    if annotate:
        for i, txt in enumerate(manips):
            if int(txt[6:]) in solids:
                ax.annotate(txt[6:], (xpos[i], ypos[i]),color = "red")
            elif int(txt[6:]) in weird:
                ax.annotate(txt[6:], (xpos[i], ypos[i]),color = "green")
            else :
                ax.annotate(txt[6:], (xpos[i], ypos[i]))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(which="both")
    if save:
        plt.savefig(loc_folder+save_loc+save+".png")
        plt.savefig(loc_folder+save_loc+save+".svg")
        if savemat:
            dico={}
            dico["xpos"]=xpos
            dico["ypos"]=ypos
            dico["xerr"]=xerr
            dico["yerr"]=yerr
            if old_data:
                dico["old_xpos"]=old_data[0]
                dico["old_ypos"]=old_data[1]
                dico["old_xerr"]=old_data[2]
                dico["old_yerr"]=old_data[3]

            scio.savemat(loc_folder+save_loc+save+".mat",dico)
    if show:
        plt.show()
    plt.close('all')



"""
## Plot creep(LC)

nice_plot(mean_lc,creep_tot,xerr=wide_lc,yerr=sigma_creep_center,
            xlabel="Loading Contrast",ylabel="Sliding between events (%)",
            annotate=True,save="sliding_VS_loading_contrast",show=show,savemat=True,
            old_data=[lc_old,ie_slip_old,lc_err_old,ie_slip_err_old])

nice_plot(mean_lc,tot_creep_center,xerr=wide_lc,yerr=sigma_creep_center,
            xlabel="Loading Contrast",ylabel="Sliding between events Center (%)",
            annotate=True,save="sliding_center_VS_loading_contrast",show=show,savemat=True)


nice_plot(mean_lc,creep_sides,xerr=wide_lc,yerr=sigma_creep_center,
            xlabel="Loading Contrast",ylabel="Sliding between events Sides (%)",
            annotate=True,save="sliding_sides_VS_loading_contrast",show=show,savemat=True)


## Plot freq(creep)
nice_plot(creep_tot, mean_freq,xerr=sigma_creep_center,yerr=wide_freq,
            xlabel="Sliding between events (%)",ylabel="Frequency (Hz)",
            annotate=True,save="frequency_VS_sliding",show=show,savemat=True,
            old_data=[ie_slip_old,freq_old,ie_slip_err_old,freq_err_old])


## Plot period(creep)
nice_plot(creep_tot, mean_dt,xerr=sigma_creep_center,yerr=sigma_dt.T,
            xlabel="Sliding between events (%)",ylabel="Period (s)",
            annotate=True,save="period_VS_sliding",show=show,savemat=True,
            old_data=None)



## Plot FD(LC)
nice_plot(mean_lc, mean_fd,xerr=wide_lc,yerr=sigma_fd,
            xlabel="Loading Contrast",ylabel="Shear force drop (kg)",
            annotate=True,save="force_drop_VS_loading_contrast",show=show,savemat=True,
            old_data=None)


## Plot FD(eps_yy_moy)
nice_plot(mean_eps_yy_ss, mean_fd,xerr=std_eps_yy_ss,yerr=sigma_fd,
            xlabel="mean sigma yy solid",ylabel="Shear force drop (kg)",
            annotate=True,save="force_drop_VS_mean_sig_ss",show=show,savemat=True,
            old_data=None)



## Plot period(eps_yy_moy)
nice_plot(mean_dt, mean_eps_yy_ss,xerr=sigma_dt,yerr=std_eps_yy_ss,
            xlabel="Drops period (s)",ylabel="mean sigma yy solid",
            annotate=True,save="period_VS_mean_sig_ss",show=show,savemat=True,
            old_data=None)


## Plot freq(LC)
nice_plot(mean_lc, mean_freq,xerr=wide_lc,yerr=wide_freq,
            xlabel="Loading Contrast",ylabel="Frequency (Hz)",
            annotate=True,save="frequency_VS_loading_contrast",show=show,savemat=True,
            old_data=[lc_old,freq_old,lc_err_old,freq_err_old])


### test plot dispertion of period(creep)

fig, ax = plt.subplots()
xpos = creep_tot
for i in range(len(freqs)):
    plt.errorbar([xpos[i] for _ in range(len(freqs[i]))],freqs[i],xerr=0,fmt="o",capsize=3)

plt.xlabel("Sliding between events (%)")
plt.ylabel("Frequency (Hz)")

for i, txt in enumerate(manips):
    if int(txt[6:]) in solids:
        ax.annotate(txt[6:], (xpos[i], mean_freq[i]),color = "red")
    elif int(txt[6:]) in weird:
        ax.annotate(txt[6:], (xpos[i], mean_freq[i]),color = "green")
    else :
        ax.annotate(txt[6:], (xpos[i], mean_freq[i]))
plt.grid(which="both")
plt.savefig(loc_folder+save_loc+"frequency_VS_sliding_scatter.png")
plt.savefig(loc_folder+save_loc+"frequency_VS_sliding_scatter.svg")

if show:
    plt.show()
plt.close('all')

"""

## Plot sanity check : integral(sigma_xy) VS Fs & yy VS Fn
from scipy.optimize import minimize


def to_min(X):
    a=0
    E,nu=X
    E=E*1e9
    for i in range(len(manips)):
        fn=-9.81*sm_fn[i]
        fs=-9.81*sm_fs[i]

        sigma_xx,sigma_yy,sigma_xy = eps_to_sigma(sm_eps_xx_mean[i],sm_eps_yy_mean[i],sm_eps_xy_mean[i],E=E,nu=nu)
        fn_bis=sigma_yy*1.5e-3
        fs_bis=sigma_xy*1.5e-3

        fn=smooth(fn,100)
        fs=smooth(fs,100)
        fn_bis=smooth(fn_bis,100)
        fs_bis=smooth(fs_bis,100)
        a+=np.sum((fn-fn_bis)**2+(fs-fs_bis)**2)
    return(a)

test=minimize(to_min,x0=np.array([3,0.3]),bounds=[[0.1,10],[0.1,0.6]])
E,nu = test["x"]
E=E*1e9

sm_sigma_xx_mean=[]
sm_sigma_yy_mean=[]
sm_sigma_xy_mean=[]

for i in range(len(sm_eps_xx_mean)):
    sxx,syy,sxy=eps_to_sigma(sm_eps_xx_mean[i],sm_eps_yy_mean[i],sm_eps_xy_mean[i],E=E,nu=nu)
    sm_sigma_xx_mean.append(sxx)
    sm_sigma_yy_mean.append(syy)
    sm_sigma_xy_mean.append(sxy)

### make the figures
"""
for i in range(len(manips)):
    plt.plot(sm_times[i],-9.81*sm_fn[i],label="Force captor measurement",alpha=.5)
    plt.plot(sm_times[i],sm_sigma_yy_mean[i]*1.5e-3,label="estimation from strain gages",alpha=.5)
    plt.xlabel("time (s)")
    plt.ylabel("Normal force (N)")
    plt.legend()
    plt.grid()
    plt.title(manips[i]+r"     $E={:.2f}\times10^9$  ;  $\nu={:.2f}$".format(E/1e9,nu))
    plt.savefig(loc_folder+manips[i]+"/Fn_VS_sigma_yy_"+ manips[i] + ".png")
    plt.savefig(loc_folder+manips[i]+"/Fn_VS_sigma_yy_"+ manips[i] + ".svg")
    plt.close('all')


    plt.plot(sm_times[i],-9.81*sm_fs[i],label="Force captor measurement",alpha=.5)
    plt.plot(sm_times[i],sm_sigma_xy_mean[i]*1.5e-3,label="estimation from strain gages",alpha=.5)
    plt.xlabel("time (s)")
    plt.ylabel("Tangential force (N)")
    plt.legend()
    plt.grid()
    plt.title(manips[i]+r"     $E={:.2f}\times10^9$  ;  $\nu={:.2f}$".format(E/1e9,nu))
    plt.savefig(loc_folder+manips[i]+"/Fs_VS_sigma_xy_"+ manips[i] + ".png")
    plt.savefig(loc_folder+manips[i]+"/Fs_VS_sigma_xy_"+ manips[i] + ".svg")
    plt.close('all')
"""





## Plot Delta eps yy entre events, par jauge, fonction de LC
"""

loading_speed_avg=np.zeros((len(eps_yy_aft),30))
loading_speed_std=np.zeros((len(eps_yy_aft),30))

x=np.array([0.005, 0.015, 0.025, 0.035, 0.045, 0.075, 0.105, 0.115, 0.125, 0.135])

for i in range(len(eps_yy_aft)):
    # get each experiment full data one by one
    aft=np.array(eps_aft[i])
    bef=np.array(eps_bef[i])
    deltat=deltats[i]
    # time is first dim, create the Delta Epsilon
    deltaeps=bef[1:,:]-aft[:-1,:]
    # obtain the slope of epsilon yy during loading
    epsilon_dot = deltaeps/deltat[:, np.newaxis]
    epsilon_dot_avg=np.average(epsilon_dot, axis=0)
    epsilon_dot_std=np.std(epsilon_dot, axis=0)
    loading_speed_avg[i]=epsilon_dot_avg
    loading_speed_std[i]=epsilon_dot_std



fig, axes = plt.subplots(1,10,sharex=True,sharey=True)

for i in range(10):
    ax=axes[i]
    ax.errorbar(mean_lc,loading_speed_avg[:,3*i+2], xerr=sigma_lc, yerr=loading_speed_std[:,3*i+2], fmt="o",capsize=3,label="x = {} mm".format(x[i]*1000))

    for j, txt in enumerate(manips):
        if int(txt[6:]) in solids:
            ax.annotate(txt[6:], (mean_lc[j], loading_speed_avg[j,3*i+2]),color = "red")
        elif int(txt[6:]) in weird:
            ax.annotate(txt[6:], (mean_lc[j], loading_speed_avg[j,3*i+2]),color = "green")
        else :
            ax.annotate(txt[6:], (mean_lc[j], loading_speed_avg[j,3*i+2]))


    #plt.xlabel("Loading contrast")
    #plt.ylabel("Local loading speed, $\dfrac{\mathrm{d}\epsilon_{yy}}{\mathrm{d}t}$ (strain/s)")
    ax.legend()
    ax.grid(which="both")
    #plt.savefig(loc_folder+save_loc+"force_drop_VS_period.png")
    #plt.savefig(loc_folder+save_loc+"force_drop_VS_period.svg")


fig.set_size_inches(20,4)
axes[0].set_ylabel("Local loading speed, $\dfrac{\mathrm{d}\epsilon_{xy}}{\mathrm{d}t}$ (strain/s)\n negative = more compressed, positive = more dilated")
axes[0].set_xlabel("Loading contrast")
#plt.tight_layout()

plt.savefig(loc_folder+save_loc+"loading_speed.png")
"""

###






































## Imports
# Science
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from scipy import interpolate

# custom file
try :
    from Python_DAQ import *
except :
    from DAQ_Python.Python_DAQ import *



solids = [14,15,16,17,18,37,38]


### load data : set up the loading

# Location

# main folder



# name of the reference file, containing unloaded signal, usually event-001
loc_file_zero = "event-001.npy"

# parameters file
loc_params="parameters.txt"

# set to True to plot the results, set to False to only save them
plot = True


# control smoothing of the data (rolling average) and starting point
roll_smooth=10
start=0




### Location of the data inside the file
# channels containing the actual strain gages
gages_channels = np.concatenate([np.arange(0,15),np.arange(16,31)])

# channels containing the normal and tangetial force
forces_channels = [32,33]

# channel containing the trigger
trigger_channel = 34



### Load data : load and create usefull variables
## Parameters
# default x, just in case there's nothing in the saved params
x=np.array([1,2,3,4,5,8,11,12,13,14])*1e-2

import os

chosen_manips = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44]

lc_per_manip = []
lc_per_manip_std = []

lc_per_manip_2 = []
lc_per_manip_std_2 = []

sigma_yy_0_tip_per_manip = []
sigma_yy_0_tip_per_manip_std = []

sigma_yy_0_tip_per_manip_2 = []
sigma_yy_0_tip_per_manip_std_2 = []

solid_per_manip = []
solid_per_manip_2 = []


start_per_event = []

n_event_per_manip = []

for j in range(0,len(chosen_manips)):

    print(chosen_manips[j])

    loc_folder_2="E:/2023-2024/2023-07-11-manips-10-voies/manip_{}/".format(chosen_manips[j])

    directory = loc_folder_2
    files = [file for file in os.listdir(directory) if file.startswith("event-") and file.endswith("_times_hand_picked.npy")]
    n_events = max(int(file.split('-')[1].split('.')[0]) for file in files)+1

    # Load the params file, and extract frequency of acquisition
    exec(load_params(loc_folder_2+loc_params))
    sampling_freq_in = clock/10
    # Create the location strings
    loc_zero=loc_folder_2+loc_file_zero
    # Number of channels
    nchannels = len(gages_channels)


    # Fast acquicition


    def comp_lc(prof):
        ss = np.mean(prof[[0,1,2,3,4,6,7,8,9]])
        gg = prof[5]
        lc = 15*(gg-ss)/(3*gg+12*ss)
        return(lc)

    lc_per_event = []
    sigma_yy_0_tip_per_event = []


    data_zero=np.load(loc_zero,allow_pickle=True)

    if chosen_manips[j]==28:
        start_avoid_2=3
    else:
        start_avoid_2=2

    solid_per_manip.append(chosen_manips[j] in solids)

    n_event_per_manip.append(0)

    for i in range(start_avoid_2,n_events):
        n_event_per_manip[-1]+=1
        solid_per_manip_2.append(chosen_manips[j] in solids)

        # name of the file / event
        loc_file="event-0{:02d}.npy".format(i)
        print(loc_file)
        loc=loc_folder_2+loc_file


        # Load data
        data=np.load(loc,allow_pickle=True)

        # smooth data
        data=smooth(data,roll_smooth)
        data=np.transpose(np.transpose(data)-np.mean(data_zero,axis=1))


        # assign specific channels to specific variables
        forces=data[forces_channels,:]
        mu = data[forces_channels[1],:].mean() / data[forces_channels[0],:].mean()
        gages = data[gages_channels]
        gages_zero = data_zero[gages_channels]
        fast_time=np.arange(len(gages[0]))/sampling_freq_in

        # create labels
        ylabels = [
        r"$\varepsilon_{{xx}}^{{{}}}$" ,
        r"$\varepsilon_{{yy}}^{{{}}}$" ,
        r"$\varepsilon_{{xy}}^{{{}}}$"
                ] * (nchannels//3)

        for i in range(len(ylabels)):
            ylabels[i]=ylabels[i].format((i+3)//3)

        converted = False

        # Convert voltage to strains or forces, and rosette to tensor
        if not converted:
            for i in range(nchannels//3):
                ch_1=gages[3*i]
                ch_2=gages[3*i+1]
                ch_3=gages[3*i+2]
                ch_1,ch_2,ch_3=voltage_to_strains(ch_1,ch_2,ch_3)
                ch_1,ch_2,ch_3=rosette_to_tensor(ch_1,ch_2,ch_3)
                gages[3*i]=ch_1
                gages[3*i+1]=ch_2
                gages[3*i+2]=ch_3

            forces = voltage_to_force(forces)

        converted=True

        indexes=np.load(loc+"_times_hand_picked.npy")

        start = np.argmin(indexes)
        start_per_event.append(start)
        temp = E*np.mean(gages[1::3,:1000],axis = 1)
        lc_per_event.append(comp_lc(temp))
        sigma_yy_0_tip_per_event.append(temp[start])


    sigma_yy_0_tip_per_event = np.array(sigma_yy_0_tip_per_event)
    lc_per_event = np.array(lc_per_event)

    lc_per_manip.append(np.mean(lc_per_event))
    lc_per_manip_std.append(np.std(lc_per_event))

    sigma_yy_0_tip_per_manip.append(np.mean(sigma_yy_0_tip_per_event))
    sigma_yy_0_tip_per_manip_std.append(np.std(sigma_yy_0_tip_per_event))

    lc_per_manip_2 += list(lc_per_event)

    sigma_yy_0_tip_per_manip_2 += list(sigma_yy_0_tip_per_event)







































###

dictototal={}

values = [creep_tot,sigma_creep_center,mean_freq,wide_freq,mean_dt,sigma_dt,mean_fd,sigma_fd,mean_lc,wide_lc,manip_num,creep_sides,creep_center,mean_eps_yy_ss,std_eps_yy_ss]
keys = ["creep_tot","sigma_creep_center","mean_freq","wide_freq","mean_dt","sigma_dt","mean_fd","sigma_fd","mean_lc","wide_lc","manip_num","creep_sides","creep_center","mean_eps_yy_ss","std_eps_yy_ss"]


for i in range(len(keys)):
    dictototal[keys[i]]=values[i]



np.save(loc_folder+"python_plots/summary_data.npy",dictototal)


dictototal_2={}


values_2 = [solids, lc_per_manip,lc_per_manip_std,lc_per_manip_2,sigma_yy_0_tip_per_manip,sigma_yy_0_tip_per_manip_std,sigma_yy_0_tip_per_manip_2,solid_per_manip,solid_per_manip_2,n_event_per_manip,start_per_event]

keys_2 = ["solids", "lc_per_manip","lc_per_manip_std","lc_per_event","sigma_yy_0_tip_per_manip","sigma_yy_0_tip_per_manip_std","sigma_yy_0_tip_per_event","solid_per_manip","solid_per_event","n_event_per_manip","start_per_event"]


for i in range(len(keys_2)):
    dictototal_2[keys_2[i]]=values_2[i]



np.save(loc_folder+"python_plots/summary_data_2.npy",dictototal_2)

























plt.close('all')






