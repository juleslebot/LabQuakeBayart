"""
This files is used to pick for each event of each experiment the moment at which
the crack is detected on each gage.
"""

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






### load data : set up the loading

# Location

# main folder
loc_folder="E:/2023-2024/2024-03-12-manip-granulareye-full/manip_1/"

# name of the file / event
loc_file="event-008.npy"

# name of the reference file, containing unloaded signal, usually event-001
loc_file_zero = "event-001.npy"

# parameters file
loc_params="parameters.txt"

# set to True to plot the results, set to False to only save them
plot = True


# control smoothing of the data (rolling average) and starting point
roll_smooth=20
start=0








### Location of the data inside the file
# channels containing the actual strain gages
gages_channels = np.concatenate([np.arange(0,15),np.arange(16,31),np.arange(32,47)])

# channels containing the normal and tangetial force
forces_channels = [15,31]

# channel containing the trigger
trigger_channel = 47



### Load data : load and create usefull variables
## Parameters
# default x, just in case there's nothing in the saved params
x=np.arange(0.01,0.16,0.01)

# Load the params file, and extract frequency of acquisition
exec(load_params(loc_folder+loc_params))
sampling_freq_in = clock/10
x=np.arange(0.01,0.16,0.01)

# Create the location strings
loc=loc_folder+loc_file
loc_zero=loc_folder+loc_file_zero

# Number of channels
nchannels = len(gages_channels)

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

# create labels
ylabels = [
  r"$\varepsilon_{{xx}}^{{{}}}$" ,
  r"$\varepsilon_{{yy}}^{{{}}}$" ,
  r"$\varepsilon_{{xy}}^{{{}}}$"
          ] * (nchannels//3)

for i in range(len(ylabels)):
    ylabels[i]=ylabels[i].format((i+3)//3)

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
    #trigger = trig_to_TTL(trigger)

converted=True



### Plot macroscopic checks

def generate_color_set(n):
    "Generate a color for every line"
    colors = []
    step = 1.0 / (n + 1)
    for i in range(n):
        r = i * step
        g = 0.5 # 2*(abs(n/2-i)/n)
        b = 1.0 - i * step
        a = 0.9
        colors.append((r, g, b, a))
    return(colors)


# slowmon forces and trigger
plt.figure()
plt.title("Fn, Fs and trigger")
plt.suptitle(loc_folder+loc_file,alpha=.2,size=5)


plt.plot(time_sm,fn_sm)
plt.plot(time_sm,fs_sm)
plt.plot(time_sm,trigger)

plt.xlabel("time (s)")
plt.grid()

plt.savefig(loc_folder+"slowmon_forces.png")
if plot :
    plt.show()
plt.close('all')



# strain profile

plt.figure()
plt.title("Strain profile (m/m)")
plt.suptitle(loc_folder+loc_file,alpha=.2,size=5)

plt.plot(x,np.mean(gages[1::3,:1000],axis=1))
plt.xlabel("position (m)")
plt.ylabel(r"$\varepsilon_{yy}$ (m/m)")
plt.grid(which="both")
plt.ylim((1.1*min(0,min(np.mean(gages[1::3,:1000],axis=1))),1.1*max(np.mean(gages[1::3,:1000],axis=1))))
plt.savefig(loc_folder+"{}_load_profile.png".format(loc_file[:-4]))
if plot :
    plt.show()
plt.close('all')





### Plot verification then pic events

# plot the actual picking graph
n_plot = nchannels//3


if n_plot//5==n_plot/5:
    shape_plot = (n_plot//5,5)
elif n_plot//4==n_plot/4:
    shape_plot = (n_plot//4,4)
else :
    shape_plot = (1,n_plot)

s_x = shape_plot[0]
s_y = shape_plot[1]

indexes = np.zeros(shape_plot)


def onpick(event):
    x_coord = event.mouseevent.xdata
    y_coord = event.mouseevent.ydata
    ax=event.mouseevent.inaxes
    row=(ax.get_subplotspec().rowspan.start)
    col=(ax.get_subplotspec().colspan.start)
    print(f'Picked point: ({x_coord:.2f}, {y_coord:.2f})')
    indexes[row,col]=x_coord


fig, axs = plt.subplots(shape_plot[0],shape_plot[1],sharex=True,sharey=False)

for i in range(nchannels//3):
    axs[i//s_y][i%s_y].plot(fast_time[start:],gages[2+3*i][start:], label=ylabels[2+3*i], picker=True, pickradius=6)
    axs[i//s_y][i%s_y].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axs[i//s_y][i%s_y].grid("both")
    axs[i//s_y][i%s_y].legend()

axs[-1][0].set_xlabel('time (s)')
fig.set_size_inches(14,8)
fig.canvas.mpl_connect('pick_event', onpick)
plt.suptitle(loc_folder+loc_file,alpha=.2,size=5)


plt.show()

plt.close('all')


indexes = indexes.reshape(n_plot)


###


fig = plt.figure

for i in range(10):
    plt.plot(fast_time[start:],gages[i][start:]-gages[i][start]+2*i/10000)

plt.show()

### quickly plot the result
fig,axs=plt.subplots(1,2)
plt.suptitle("Rupture propagation")


axs[0].plot(1000*indexes,x,marker="+",linestyle="-.",linewidth=.5)
axs[0].set_ylabel("Position (m)")
axs[0].set_xlabel("Time (ms)")
axs[0].grid(which="both")

axs[1].plot(x,1000*indexes,marker="+",linestyle="-.",linewidth=.5)
axs[1].set_xlabel("Position (m)")
axs[1].set_ylabel("Time (ms)")
axs[1].grid(which="both")
plt.suptitle(loc_folder+loc_file,alpha=.2,size=5)


plt.tight_layout()
plt.show()



### Plot the propagation

E=2.78e9

n_plot = nchannels//3

fig, axs = plt.subplots(n_plot,1,sharex=True,sharey=False)

for i in range(n_plot):
    axs[i].plot(1000*fast_time[start:]-1000*fast_time.mean(),-(gages[3*(i+1)-1][start:]-gages[3*(i+1)-1][0:10000].mean())*E*1e-6, label="$\Delta\sigma_{{xy}}^{{{}}}$, $x={{{}}}$".format(i+1,x[i]))
    axs[i].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axs[i].grid("both")
    #plt.ticklabel_format(style='plain', axis='y')
    #plt.yticks([0,-0.25])
    #plt.ylim([-0.35,0.15])
    #axs[i].legend(loc='upper right')
    #axs[i].axvline(1000*indexes[i]-1000*fast_time.mean(),color='k',linestyle="dashed")
    #axs[i].axhline(0,color='grey',linestyle="solid",alpha=.5,linewidth=2)

a,b = min(indexes),max(indexes)
a,b=a-(b-a)/2-fast_time.mean(),b+(b-a)/2-fast_time.mean()
axs[-1].set_xlim([1000*a,1000*b])
axs[-1].set_xlabel('time (ms)')
axs[2].set_ylabel('$\sigma_{xy}$ (MPa)')

fig.set_size_inches((7,9))
#plt.savefig(loc_folder+loc_file+"propagation_verticale.png")
plt.tight_layout
plt.suptitle(loc_folder+loc_file,alpha=.2,size=5)

###
if plot :
    plt.show()
plt.close('all')



### Print times and speeds (xy)
print(r"µ={:.3f}".format(mu))

times=indexes

print(np.diff(x)/np.diff(times))

### save them

np.save(loc+"_times_hand_picked.npy",indexes)

### pick them up

indexes=np.load(loc+"_times_hand_picked.npy")

### Fit propagation

from scipy.odr import Model, RealData, ODR

def linear_func(p, x):
    return p[0]*x + p[1]

linear_model = Model(linear_func)





x_err = np.ones(indexes.shape)*1e-3

times=indexes-np.mean(indexes)


y=[times[i] for i in range(len(indexes))]
# y=[np.mean(times[:,i]) for i in range(5)]
#y_err=[np.std(times[:,i]) for i in range(5)]
y_err=[1e-5 for i in range(len(indexes))]


# determine speed
datar = RealData(y, x, sx=y_err, sy=x_err)
odr = ODR(datar, linear_model, beta0=[1, 0])
out = odr.run()
popt = out.beta
perr = out.sd_beta
v=popt[0]
sigmav=perr[0]

# plot
#plt.scatter(x,times[0],label=r"$\varepsilon_{xx}$")
#plt.scatter(x,times[1],label=r"$\varepsilon_{yy}$")
plt.scatter(x,times,label=r"$\varepsilon_{xy}$")
plt.errorbar(x,y,yerr=y_err,xerr=x_err)#,label=r"All $\varepsilon$")


datar = RealData(x, y, sx=x_err, sy=y_err)
odr = ODR(datar, linear_model, beta0=[1, 0])
out = odr.run()
popt = out.beta
perr = out.sd_beta
plt.plot(x, linear_func(popt, x), 'r-', label='Linear fit, $v={:.0f}\pm{:.0f}$ m/s'.format(v,sigmav))

plt.xlabel("Position (m)")
plt.ylabel("Temps de passage (s)")

plt.grid(which="both")
plt.legend()
plt.suptitle(loc_folder+loc_file,alpha=.2,size=5)


plt.savefig(loc+"_propagation.png",dpi=600)
if plot :
    plt.show()
plt.close('all')





