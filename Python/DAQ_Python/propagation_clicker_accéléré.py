## Imports
# Science
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from scipy import interpolate
import os
import fnmatch

try :
    from Python_DAQ import *
except :
    from DAQ_Python.Python_DAQ import *



def voltage_to_strains(ch1,ch2,ch3):
    side=lambda x: V_to_strain(x,amp=495,G=1.79,i_0=0.0017,R=350)
    center=lambda x: -V_to_strain(x,amp=495,G=1.86,i_0=0.0017,R=350)
    ch1=side(ch1)
    ch2=center(ch2)
    ch3=side(ch3)
    return(ch1,ch2,ch3)

def voltage_to_force(ch):
    # correct sign
    for i in range(len(ch)):
        ch[i]=ch[i]*np.median(np.sign(ch[i]))
    return(500/3*ch)

def list_files_with_pattern(directory, pattern):
    matching_files = []
    for file_name in os.listdir(directory):
        if fnmatch.fnmatch(file_name, f'*{pattern}*'):
            matching_files.append(file_name)

    return(matching_files)

def onpick(event):
    x_coord = event.mouseevent.xdata
    y_coord = event.mouseevent.ydata
    ax=event.mouseevent.inaxes
    row=(ax.get_subplotspec().rowspan.start)
    col=(ax.get_subplotspec().colspan.start)
    print(f'Picked point: ({x_coord:.2f}, {y_coord:.2f})')
    picked_starts[-1][row]=not(picked_starts[-1][row])
    print(picked_starts[-1])




def temp_nice_plot(y_min,x, ax = plt.gca() ) :
    """
    plots a nice representation of the bloc
    """
    ylim=ax.get_ylim()
    dilat = ylim[1]-ylim[0]
    ax.set_ylim((y_min-0.1*dilat,ylim[1]))
    import matplotlib.patches as patches
    arc_radius = 0.05*dilat
    arc_center_x = 0.075
    arc_center_y = y_min
    start_angle = 0
    end_angle = 180
    arc_patch = patches.Arc((arc_center_x, arc_center_y), width=0.03, height=2*arc_radius, angle=0,
                            theta1=start_angle, theta2=end_angle, color='r', linewidth=2)
    ax.add_patch(arc_patch)

    # lines

    line1 = patches.ConnectionPatch((0, y_min), (0.06, y_min), "data", "data", edgecolor="r", linewidth=2, arrowstyle="-")
    ax.add_patch(line1)
    line2 = patches.ConnectionPatch((0.09, y_min), (0.15, y_min), "data", "data", edgecolor="r", linewidth=2, arrowstyle="-")
    ax.add_patch(line2)
    line3 = patches.ConnectionPatch((0, y_min), (0, y_min+0.25*dilat), "data", "data", edgecolor="r", linewidth=2, arrowstyle="-")
    ax.add_patch(line3)
    line4 = patches.ConnectionPatch((0.15, y_min), (0.15, y_min+0.25*dilat), "data", "data", edgecolor="r", linewidth=2, arrowstyle="-")
    ax.add_patch(line4)

    x_not = [0.005+0.01*i for i in range(15)]
    for xi in x_not:
        square_patch = patches.Rectangle((xi-0.002, y_min+0.07*dilat), 0.004, 0.04*dilat, color='grey',alpha=.3)
        ax.add_patch(square_patch)

    for xi in x:
        square_patch = patches.Rectangle((xi-0.002, y_min+0.07*dilat), 0.004, 0.04*dilat, color='r',alpha=1)
        ax.add_patch(square_patch)

    ax.set_xlim((-0.01,0.16))


### load data : set up the loading
# Location
loc_folder_main="D:/Users/Manips/Documents/DATA/FRICS/2023/2023-07-11-manips-10-voies/"
loc_file_zero = "event-001.npy"
loc_params="parameters.txt"
to_remove = [7,19,20,21,22,23,24,25,32,33,41]
gages_channels = np.concatenate([np.arange(0,15),np.arange(16,31)])
forces_channels = [32,33]
trigger_channel = 34
plot = False
# Parameters
roll_smooth=31
start=0
picked_starts = []
loading_contrast = []
manips=["manip_{}".format(i) for i in range(1,45)]


resume= 0


if resume:
    picked_starts=list(np.load(loc_folder_main+"picked_starts.npy"))
    loading_contrast=list(np.load(loc_folder_main+"loading_contrast.npy"))
    manips=manips[resume:]



for i in to_remove:
    try:
        manips.remove("manip_{}".format(i))
    except:
        print("{} not in list".format(i))

for manip in manips:

    loc_folder=loc_folder_main+manip+"/"
    events=list_files_with_pattern(loc_folder,"event-")
    events.remove("event-001.npy")
    events=sorted([a for a in events if len(a)==13])
    for loc_file in events:
        picked_starts.append([False for _ in range(10)])
        print(manip)
        print(loc_file)



        # Load data : load and create usefull variables
        # Load the params file
        # just in case there's nothing in the saved params
        x=np.array([1,2,3,4,5,8,11,12,13,14])*1e-2
        exec(load_params(loc_folder+loc_params))
        sampling_freq_in = clock/10

        # Create the location strings
        loc=loc_folder+loc_file
        loc_zero=loc_folder+loc_file_zero

        # Number of channels
        nchannels = len(gages_channels)


        # Load data
        data=np.load(loc,allow_pickle=True)
        data_zero=np.load(loc_zero,allow_pickle=True)

        # smooth data
        data=smooth(data,roll_smooth)
        data=np.transpose(np.transpose(data)-np.mean(data_zero,axis=1))



        forces=data[forces_channels,:]

        mu = data[forces_channels[1],:].mean() / data[forces_channels[0],:].mean()
        gages = data[gages_channels]
        gages_zero = data_zero[gages_channels]
        time=np.arange(len(gages[0]))/sampling_freq_in

        # Convert voltage to strains or forces, and rosette to tensor

        for i in range(nchannels//3):
            ch_1=gages[3*i]
            ch_2=gages[3*i+1]
            ch_3=gages[3*i+2]
            ch_1,ch_2,ch_3=voltage_to_strains(ch_1,ch_2,ch_3)
            ch_1,ch_2,ch_3=rosette_to_tensor(ch_1,ch_2,ch_3)
            gages[3*i]=ch_1
            gages[3*i+1]=ch_2
            gages[3*i+2]=ch_3

        for i in range(nchannels//3):
            ch_1=gages_zero[3*i]
            ch_2=gages_zero[3*i+1]
            ch_3=gages_zero[3*i+2]
            ch_1,ch_2,ch_3=voltage_to_strains(ch_1,ch_2,ch_3)
            ch_1,ch_2,ch_3=rosette_to_tensor(ch_1,ch_2,ch_3)
            gages_zero[3*i]=ch_1
            gages_zero[3*i+1]=ch_2
            gages_zero[3*i+2]=ch_3

        forces = voltage_to_force(forces)






        avg_before = np.mean(gages[:,:1000],axis=-1)
        avg_after  = np.mean(gages[:,-1000:],axis=-1)
        eps_yy=avg_before[1::3]
        loading_contrast.append( (eps_yy[5] - np.mean(eps_yy[[i for i in range(10) if i!=5]])) /np.mean(eps_yy) )










        #Plot the propagation

        E=2.78e9


        fig, axs = plt.subplots(10,1,sharex=True,sharey=True)

        for i in range(10):
            toplot=(gages[3*(i+1)-1][start:]-gages[3*(i+1)-1][0:10000].mean())*E*1e-6
            axs[i].plot(1000*time[start:]-1000*time.mean(),toplot, label="$\Delta\sigma_{{xy}}^{{{}}}$, $x={{{}}}$".format(i+1,x[i]),picker=True, pickradius=1000)
            axs[i].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            axs[i].grid("both")
            axs[i].legend(loc='upper right')
            axs[i].plot(1000*time[start:]-1000*time.mean(),np.ones(time[start:].shape)*np.mean(toplot[:10000]))


        axs[-1].set_xlim([-0.2,0.1])
        axs[-1].set_xlabel('time (ms)')
        axs[2].set_ylabel('$\sigma_{xy}$ (MPa)')

        plt.tight_layout
        plt.suptitle(loc_folder+loc_file,alpha=.2,size=5)
        fig.canvas.mpl_connect('pick_event', onpick)

        plt.show()

### save

if True :
    picked_starts=np.array(picked_starts)
    loading_contrast=np.array(loading_contrast)
    np.save(loc_folder_main+"picked_starts",picked_starts)
    np.save(loc_folder_main+"loading_contrast",loading_contrast)



###
picked_starts=np.load(loc_folder_main+"picked_starts.npy")
loading_contrast=np.load(loc_folder_main+"loading_contrast.npy")


picked = np.array([a/a.sum() for a in picked_starts])



###
x=[0.005,0.015,0.025,0.035,0.045,0.075,0.105,0.115,0.125,0.135]








bin_width=.5
bin_variable=loading_contrast
x_axis=x


max_lc = np.ceil(max(bin_variable)/bin_width)*bin_width
min_lc = np.floor(min(bin_variable)/bin_width)*bin_width
n_bin =  int(np.ceil((max_lc-min_lc)/bin_width))

# sort the values in the bins
hists = []
for i in range(n_bin):
    hists.append(np.array(np.array([picked[k]
                        for k in range(len(picked))
                        if  bin_variable[k]<min_lc+bin_width*(i+1)
                        and bin_variable[k]>=min_lc+bin_width*i
                    ]).sum(axis=0)))

hists=np.array(hists)








### A FINIR
fig, axes = plt.subplots(nrows=int(np.ceil(n_bin/2)), ncols=2, sharey=True)

fig.subplots_adjust(hspace=0.5)
fig.subplots_adjust(hspace=0.5)
for i, ax in enumerate(axes.flat):
    if i<n_bin:
        y_min=-2
        ax.bar(x_axis, hists[i],width=0.8*min(np.diff(x_axis)))
        ax.set_title("{:.1f} $\leq$ {} < {:.1f}".format(min_lc+i*bin_width,"LC",min_lc+(i+1)*bin_width))
        temp_nice_plot(y_min,x_axis,ax=ax)
        ax.grid(which="both")

fig.suptitle("Jauge de départ en fonction du Loading Contrast ")
axes[0][0].set_ylabel("Occurences (pour chaque bin)",fontsize="5")



plt.savefig("D:/Users/Manips/Documents/DATA/FRICS/2023/2023-07-11-manips-10-voies/histograms/start_picked_quad_hist.png",dpi=600)
plt.savefig("D:/Users/Manips/Documents/DATA/FRICS/2023/2023-07-11-manips-10-voies/histograms/start_picked_quad_hist.svg")
plt.show()












