import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import fnmatch
import scipy.io as scio
import glob


try :
    from Python_DAQ import *
except :
    from DAQ_Python.Python_DAQ import *



def list_files_with_pattern(directory, pattern):
    matching_files = []
    for file_name in os.listdir(directory):
        if fnmatch.fnmatch(file_name, f'*{pattern}*'):
            matching_files.append(file_name)

    return(matching_files)


## Data location

loc_folder="D:/Users/Manips/Documents/DATA/FRICS/2023/2023-07-11-manips-10-voies/"
loc_manip = "manip_{}/"
save_loc = "python_plots/"
gages_channels = np.concatenate([np.arange(0,15),np.arange(16,31)])
forces_channels = np.array([32,33])
nchannels = len(gages_channels)
trigger_channel = 34
## load daq data
loading_contrast = []
force_drop = []
fn = []
fs = []
sm_trigs = []
sm_times = []

to_remove = [7,19,20,21,22,23,24,25,32,33]
n_manip = 38
solids = [ 14,15,16,17,18,37,38 ]
weird = [33]
manips = ["manip_{}".format(i) for i in range(1,n_manip+1) if i not in to_remove]





for manip in manips:
    directory_path = loc_folder + manip + "/"
    sm_trigs.append(np.load(directory_path+"slowmon.npy")[trigger_channel])
    sm_times.append(np.load(directory_path+"slowmon_time.npy"))
    lc_temp = []
    force_drops = []
    fns = []
    fss = []
    pattern = 'event-[0-9][0-9][0-9].npy'
    matching_files = glob.glob(f"{directory_path}/{pattern}")
    _=matching_files.pop(0)
    for i in range(len(matching_files)):
        matching_files[i]=matching_files[i][-13:]
    for file in matching_files :
        # gages
        daqdata = np.load(directory_path+file[:13])
        gages  = daqdata[gages_channels]
        # forces
        forces = daqdata[forces_channels]*500/3
        forces[0]=forces[0]*(1 if np.mean(forces[0])>0 else -1)
        forces[1]=forces[1]*(1 if np.mean(forces[1])>0 else -1)
        fns.append(np.mean(forces[0][:10000]))
        fss.append(np.mean(forces[1][:10000]))
        force_drops.append(np.mean(forces[1][:10000])-np.mean(forces[1][-10000:]))
        #plt.plot(forces[1])
        #plt.axhline(fss[-1])
        #plt.axhline(fss[-1]-force_drops[-1])
        #plt.show()
        #gages  = np.load(directory_path+file)[gages_channels]
        gages_zero = np.load(directory_path+"event-001.npy")[gages_channels]
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
        eps_yy=avg_before[1::3]
        #plt.plot(eps_yy);plt.show()
        med = np.median(eps_yy[[i for i in range(10) if i!=5]])
        #med = np.median(eps_yy[[1,4,6,9]])
        lc_temp.append( (eps_yy[5] - med) / med )
    fn.append(fns)
    fs.append(fss)
    force_drop.append(force_drops)
    loading_contrast.append(lc_temp)

## load matlab data
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


## extract frequency
freqs=[]
deltats=[]
# for i in range(len(sm_trigs)):
#     # sm_trigs[i]=sm_trigs[i][10000:]
#     sm_trigs[i]=sm_trigs[i]>3
#     sm_trigs[i]=np.diff(sm_trigs[i])>0.5
#     sm_trigs[i]=np.where(sm_trigs[i])[0]
#     timings = sm_times[i][10000+sm_trigs[i]]
#     deltat = np.diff(timings)
#     freq = 1/deltat
#     freq=freq[freq<2]
#     freq=freq[np.where(freq<3*np.median(freq))]
#     freqs.append(freq)

for i in range(len(sm_trigs)):
    # sm_trigs[i]=sm_trigs[i][10000:]
    sm_trigs[i]=sm_trigs[i]>3
    stop = len(sm_trigs[i])
    j=10000
    timings=[]
    while j < stop:
        if sm_trigs[i][j] :
            k=0
            while j+k<stop and sm_trigs[i][j+k]:
                k+=1
            if k>5 :
                timings.append(j)
            j=j+int(10*k)
        else :
            j+=1
    #print(i)
    #plt.plot(sm_trigs[i])
    #for a in timings:
    #    plt.axvline(x=a)
    #plt.show()
    timings = sm_times[i][timings]
    deltat = np.diff(timings)
    freq = 1/deltat
    #freq=freq[freq<2]
    #freq=freq[np.where(freq<3*np.median(freq))]
    freqs.append(freq)
    deltats.append(deltat)



mean_freq = np.array([np.mean(freq) for freq in freqs])
wide_freq = np.array([[np.quantile(freq,.25),np.quantile(freq,.75)] for freq in freqs])
sigma_freq = np.array([np.std(freq) for freq in freqs])
mean_dt = np.array([np.mean(dt) for dt in deltats])
sigma_dt = np.array([np.std(dt) for dt in deltats])

## extract LC
mean_lc=np.array([np.mean(lc) for lc in loading_contrast])
sigma_lc=np.array([np.std(lc) for lc in loading_contrast])
mean_fd=np.array([np.mean(fd) for fd in force_drop])
sigma_fd=np.array([np.std(fd) for fd in force_drop])

## extract inter event slip
tot_creep_left = np.array([np.median(c[-5:]) for c in creep_left])
tot_creep_right = np.array([np.median(c[-5:]) for c in creep_right])
tot_creep_center = np.array([np.median(c[-5:]) for c in creep_center])
sigma_creep_left = np.array([np.std(c[-5:]) for c in creep_left])
sigma_creep_right = np.array([np.std(c[-5:]) for c in creep_right])
sigma_creep_center = np.array([np.std(c[-5:]) for c in creep_center])

## Plot creep(LC)
fig, ax = plt.subplots()
y=tot_creep_center-(tot_creep_left+tot_creep_right)/2
plt.errorbar(mean_lc,y, xerr=sigma_lc,yerr=sigma_creep_center,fmt="o",capsize=3,label="new data")

for i, txt in enumerate(manips):
    if int(txt[6:]) in solids:
        ax.annotate(txt[6:], (mean_lc[i], y[i]),color = "red")
    elif int(txt[6:]) in weird:
        ax.annotate(txt[6:], (mean_lc[i], y[i]),color = "green")
    else :
        ax.annotate(txt[6:], (mean_lc[i], y[i]))

# Old data
c=[-1,-0.9,-0.9,-0.8,-0.7,-0.3,-0.2,0.9,2.01,2.35,2.95]
cpm=[0.01,0.01,0.01,0.1,0.05,0.5,0.2,0.6,0.2,0.3,0.2]
s=[3,5,2.5,-0.05,3.5,4.95,7,10,12,16.5,22]
spm=[2,2,2,2,2,2,2,2,2,2,2]
plt.errorbar(c,s,xerr=cpm,yerr=spm,fmt=' ',capsize=3, marker='o',alpha=.4,label="old data")
plt.xlabel("Loading Contrast")
plt.ylabel("Sliding between events (µm)")
plt.legend()
plt.grid(which="both")
plt.savefig(loc_folder+save_loc+"sliding_VS_loading_contrast.png")
plt.savefig(loc_folder+save_loc+"sliding_VS_loading_contrast.svg")
plt.show()

## Plot freq(creep)
fig, ax = plt.subplots()
x_pos = tot_creep_center-(tot_creep_left+tot_creep_right)/2
plt.errorbar(x_pos, mean_freq,xerr=sigma_creep_center,yerr=wide_freq.T, fmt="o",capsize=3,label="new data")
plt.xlabel("Sliding between events (%)")
plt.ylabel("Frequency (Hz)")
plt.legend()



for i, txt in enumerate(manips):
    if int(txt[6:]) in solids:
        ax.annotate(txt[6:], (x_pos[i], mean_freq[i]),color = "red")
    elif int(txt[6:]) in weird:
        ax.annotate(txt[6:], (x_pos[i], mean_freq[i]),color = "green")
    else :
        ax.annotate(txt[6:], (x_pos[i], mean_freq[i]))
plt.grid(which="both")
plt.savefig(loc_folder+save_loc+"frequency_VS_sliding.png")
plt.savefig(loc_folder+save_loc+"frequency_VS_sliding.svg")

plt.show()



### test

fig, ax = plt.subplots()
x_pos = tot_creep_center-(tot_creep_left+tot_creep_right)/2
for i in range(len(freqs)):
    plt.errorbar([x_pos[i] for _ in range(len(freqs[i]))],
     freqs[i],
     xerr=sigma_creep_center[i],
     fmt="o",
     capsize=3)


plt.xlabel("Sliding between events (%)")
plt.ylabel("Frequency (Hz)")
plt.legend()



for i, txt in enumerate(manips):
    if int(txt[6:]) in solids:
        ax.annotate(txt[6:], (x_pos[i], mean_freq[i]),color = "red")
    elif int(txt[6:]) in weird:
        ax.annotate(txt[6:], (x_pos[i], mean_freq[i]),color = "green")
    else :
        ax.annotate(txt[6:], (x_pos[i], mean_freq[i]))
plt.grid(which="both")
#plt.savefig(loc_folder+save_loc+"frequency_VS_sliding.png")
#plt.savefig(loc_folder+save_loc+"frequency_VS_sliding.svg")

plt.show()


## Plot FD(LC)
fig, ax = plt.subplots()
plt.errorbar(mean_lc,mean_fd, xerr=sigma_lc,yerr=sigma_fd,fmt="o",capsize=3,label="new data")

for i, txt in enumerate(manips):
    if int(txt[6:]) in solids:
        ax.annotate(txt[6:], (mean_lc[i], mean_fd[i]),color = "red")
    elif int(txt[6:]) in weird:
        ax.annotate(txt[6:], (mean_lc[i], mean_fd[i]),color = "green")
    else :
        ax.annotate(txt[6:], (mean_lc[i], mean_fd[i]))


plt.xlabel("Loading Contrast")
plt.ylabel("Shear force drop (kg)")
plt.legend()
plt.grid(which="both")
plt.savefig(loc_folder+save_loc+"force_drop_VS_loading_contrast.png")
plt.savefig(loc_folder+save_loc+"force_drop_VS_loading_contrast.svg")

plt.show()

## Plot deltat(fd)
fig, ax = plt.subplots()
plt.errorbar(mean_dt, mean_fd, xerr=sigma_dt, yerr=sigma_fd, fmt="o",capsize=3,label="new data")

for i, txt in enumerate(manips):
    if int(txt[6:]) in solids:
        ax.annotate(txt[6:], (mean_dt[i], mean_fd[i]),color = "red")
    elif int(txt[6:]) in weird:
        ax.annotate(txt[6:], (mean_dt[i], mean_fd[i]),color = "green")
    else :
        ax.annotate(txt[6:], (mean_dt[i], mean_fd[i]))


plt.xlabel("Drops period (s)")
plt.ylabel("Shear force drop (kg)")
plt.legend()
plt.grid(which="both")
plt.savefig(loc_folder+save_loc+"force_drop_VS_period.png")
plt.savefig(loc_folder+save_loc+"force_drop_VS_period.svg")

plt.show()


