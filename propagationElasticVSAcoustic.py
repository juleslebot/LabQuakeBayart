"""
This files is used to pick for each event of each experiment the moment at which
the crack is detected on each gage.
"""

## Imports
# Science
import matplotlib.pyplot as plt
import numpy as np
import os

# custom file
try :
    from Python_DAQ import *
except :
    from DAQ_Python.Python_DAQ import *


loc_folder="E:/2023-2024/2024-05-13-acoustique/02-strain-acoustique/"
loc_file="event-005.npy"
loc_save=loc_folder+loc_file[:-4]+"/"
loc_file_zero = "event-001.npy"
loc_params="parameters.txt"
loc=loc_folder+loc_file
loc_zero=loc_folder+loc_file_zero


# control smoothing of the data (rolling average) and starting point
roll_smooth=1
start=0

## gages of interest and other channels
# if you want them all :
gages_true_number = np.arange(1,61,1)
gages_channels = gauge_number_to_channel(gages_true_number)
nchannels = len(gages_channels)
forces_channels = [15,31]
trigger_channel = 47

ylabels = [
  r"$\varepsilon_{{xx}}^{{{}}}$" ,
  r"$\varepsilon_{{yy}}^{{{}}}$" ,
  r"$\varepsilon_{{xy}}^{{{}}}$"
          ] * (nchannels//3)

for i in range(len(ylabels)):
    ylabels[i]=ylabels[i].format((i+3)//3)


## Load Parameters
# if x is not set in the parameters, set a default x :
x=np.linspace(0.012,0.145,20)

exec(load_params(loc_folder+loc_params))

sampling_freq_in = clock/10

n_plot = nchannels//3


## Fast acquicition
# Load data
data=np.load(loc,allow_pickle=True)
data_zero=np.load(loc_zero,allow_pickle=True)

# smooth data
data=smooth(data,roll_smooth)

# assign specific channels to specific variables
forces=data[forces_channels,:]
mu = data[forces_channels[1],:].mean() / data[forces_channels[0],:].mean()
gages = data[gages_channels]
gages_zero = data_zero[gages_channels]
gages=np.transpose(np.transpose(gages)-np.mean(gages_zero,axis=1))

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

    eps_xx_sm = gages_sm[0::3]
    eps_yy_sm = gages_sm[1::3]
    eps_xy_sm = gages_sm[2::3]

    sigma_sm = np.zeros_like(gages_sm)
    for i in range(nchannels//3):
        sigma_sm[3*i], sigma_sm[3*i+1], sigma_sm[3*i+2] = eps_to_sigma( eps_xx_sm[i],
                          eps_yy_sm[i],
                          eps_xy_sm[i],
                          E=E,nu=nu )

    sigma = np.zeros_like(gages)
    for i in range(nchannels//3):
        sigma[3*i], sigma[3*i+1], sigma[3*i+2] = eps_to_sigma( gages[3*i],
                          gages[3*i+1],
                          gages[3*i+2],
                          E=E,nu=nu )



    forces = voltage_to_force(forces)
    forces_sm=voltage_to_force(forces_sm)
    fn_sm = voltage_to_force(fn_sm)
    fs_sm = voltage_to_force(fs_sm)

converted=True


### Pick timings
# define geometry of the plot
n_plot = nchannels//3
if n_plot//5==n_plot/5:
    shape_plot = (n_plot//5,5)
elif n_plot//4==n_plot/4:
    shape_plot = (n_plot//4,4)
else :
    shape_plot = (1,n_plot)

s_x = shape_plot[0]
s_y = shape_plot[1]

# create the click picker
indexes = np.zeros(shape_plot)

def onpick(event):
    x_coord = event.mouseevent.xdata
    y_coord = event.mouseevent.ydata
    ax=event.mouseevent.inaxes
    row=(ax.get_subplotspec().rowspan.start)
    col=(ax.get_subplotspec().colspan.start)
    print(r'Picked point: ({:.2f}, {:.2f})'.format(1000*x_coord,1000*y_coord))
    indexes[row,col]=x_coord


# create the figure
fig, axs = plt.subplots(shape_plot[0],shape_plot[1],
                        sharex=True,sharey=False,
                        constrained_layout = True)

# plot
for i in range(nchannels//3):
    axs[i//s_y][i%s_y].plot(fast_time[start:],gages[2+3*i][start:], label=ylabels[2+3*i],
                            picker=True, pickradius=6)
    axs[i//s_y][i%s_y].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axs[i//s_y][i%s_y].grid("both")
    axs[i//s_y][i%s_y].legend()

axs[-1][0].set_xlabel('time (s)')
axs[0,0].title.set_text("click to pick")
fig.set_size_inches(14,8)
fig.canvas.mpl_connect('pick_event', onpick)
plt.suptitle(loc_folder+loc_file,alpha=.2,size=5)
plt.show()
plt.close('all')



indexes = indexes.reshape(n_plot)

### Save picked times
np.save(loc+"_times_hand_picked.npy",indexes)



### Load already picked times
indexes=np.load(loc+"_times_hand_picked.npy")



### plotting propagation and measuring speeds
# sort the data
ind = np.argsort(x)
x_sort = x[ind]
indexes_sort = indexes[ind]


# define the clicker
speed_temp = np.array([0.,0.])

def onpick(event):
    x_coord = event.mouseevent.xdata
    y_coord = event.mouseevent.ydata
    ax=event.mouseevent.inaxes
    col=(ax.get_subplotspec().colspan.start)
    if speed_temp[0] == 0 and speed_temp[1] == 0:
        speed_temp[0] = x_coord
        speed_temp[1] = y_coord
        print("point picked")
    else:
        speed = np.abs((speed_temp[0]-x_coord)/(speed_temp[1]-y_coord))
        if col == 0:
            print(r"measured speed : {:.1f} m/s".format(1000*speed))
        else:
            print(r"measured speed : {:.1f} m/s".format(1000/speed))
        speed_temp[0] = 0
        speed_temp[1] = 0

# create the plot
fig,axs=plt.subplots(1,2)
plt.suptitle("Rupture propagation")
plt.suptitle(loc_folder+loc_file,alpha=.2,size=5)
plt.title("clic to measure")

# plot the data
axs[0].plot(1000*indexes_sort,x_sort,marker="+",linestyle="-.",linewidth=.5,picker=True, pickradius=6)
axs[0].set_ylabel("Position (m)")
axs[0].set_xlabel("Time (ms)")
axs[0].grid(which="both")

axs[1].plot(x_sort,1000*indexes_sort,marker="+",linestyle="-.",linewidth=.5,picker=True, pickradius=6)
axs[1].set_xlabel("Position (m)")
axs[1].set_ylabel("Time (ms)")
axs[1].grid(which="both")

fig.canvas.mpl_connect('pick_event', onpick)
plt.tight_layout()
plt.show()






### Plot the propagation
# E and nu are already imported

newtime = 1000*fast_time[start:]-1000*fast_time.mean()
delta_sigma = np.transpose(np.transpose(sigma)-sigma[:,:1000].mean(axis=-1))
delta_eps = np.transpose(np.transpose(gages)-gages[:,:1000].mean(axis=-1))
labelstring = "$\Delta\epsilon_{{xy}}^{{{}}}$, $x={{{}}}$"
ind = np.argsort(x)


fig, axs = plt.subplots(n_plot,1,sharex=True,sharey=True)
for i in range(n_plot):
    axs[i].plot(newtime, delta_eps[3*ind[i]+2], label=labelstring.format(i+1,x[ind[i]]))
    axs[i].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axs[i].grid("both")
    #plt.ticklabel_format(style='plain', axis='y')
    #plt.yticks([0,-0.25])
    #plt.ylim([-0.35,0.15])
    #axs[i].legend(loc='upper right')
    axs[i].axvline(1000*indexes[ind[i]]-1000*fast_time.mean(),color='k',linestyle="dashed")
    #axs[i].axhline(0,color='grey',linestyle="solid",alpha=.5,linewidth=2)

a,b = min(indexes),max(indexes)
a,b=a-(b-a)/2-fast_time.mean(),b+(b-a)/2-fast_time.mean()
axs[-1].set_xlim([2000*a,2000*b])

axs[-1].set_xlabel('time (ms)')
axs[2].set_ylabel('$\epsilon_{xy}$')

fig.set_size_inches((7,9))
#plt.savefig(loc_folder+loc_file+"propagation_verticale.png")
plt.subplots_adjust(hspace=0,top=0.98,bottom=0.05)
plt.suptitle(loc_folder+loc_file,alpha=.2,size=5)
plt.show()





## Save data for fit

location_gages_height="E:/2023-2024/2024-02-06-solid-solid-full/hauteur.txt"

height = np.loadtxt(location_gages_height)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

test=[find_nearest(fast_time,i) for i in indexes]

dictosave={}
dictosave["sigma"]=sigma
dictosave["epsilon"]=delta_eps
dictosave["newtime"]=newtime/1000
dictosave["ind"]=ind
dictosave["indexes"]=test
dictosave["height"]=height
dictosave["x"]=x



isExist = os.path.exists(loc_save)
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(loc_save)
   print("The new directory is created!")

np.save(loc_save+"data_test.npy",dictosave)

#########################

path = "E:/2023-2024/2024-05-13-acoustique/02-strain-acoustique/"
filename = 'Sampling.tdms'
filepath = path+filename
tdms_file = TdmsFile.read(filepath)
group = tdms_file['Analog inputs']
x_sensor = [0.045,0.075,0.105]


ch1 = group['Voie 2'][:]
ch2 = group['Voie 3'][:]
ch3 = group['Voie 1'][:]

index = []
for n in range((len(ch1)//1000000)-1):
    if np.max(ch1[n*1000000:(n+1)*1000000]) > 1000:
        index.append([n*1000000,(n+1)*1000000])
if np.max(ch1[(n+1)*1000000:])> 1000:
          index.append([(n+1)*1000000,(n+2)*1000000])


for i,ind in enumerate(index):
    print(i,' : ',ind)
print('manual : choose manualy')
print('all : all channel')

ind = input('Choose index : ')
if ind == 'manual':
    Nstart = int(input('Nstart = '))
    Nend = int(input('Nend = '))
if ind == 'all':
    Nstart = 0
    Nend = len(ch1)
else:
    ind = int(ind)
    Nstart = index[ind][0]
    Nend = index[ind][1]

Npoint = Nend - Nstart

ch1 = group['Voie 2'][Nstart:Nend]
ch2 = group['Voie 3'][Nstart:Nend]
ch3 = group['Voie 1'][Nstart:Nend]

fs = 10e+6 # sampling frequency (Hz)
time = np.linspace(0,Npoint/fs,Npoint)

ch1 = ch1/max(ch1)
ch2 = ch2/max(ch2)
ch3 = ch3/max(ch3)



### Pick timings


def onpick(event):
    x_coord = event.mouseevent.xdata
    y_coord = event.mouseevent.ydata
    ax=event.mouseevent.inaxes
    col=(ax.get_subplotspec().rowspan.start)
    print(r'Picked point: ({:.2f}, {:.2f})'.format(1000*x_coord,1000*y_coord))
    indexes[col]=x_coord


# create the figure
fig, axs = plt.subplots(3,1,sharex=True,sharey=True,constrained_layout = True)
indexes = np.zeros((3))

# plot

axs[0].plot(time,ch1, label='capteur 11',picker=True, pickradius=6)
axs[0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
axs[0].grid("both")
axs[0].legend()
axs[1].plot(time,ch2, label='capteur 21',picker=True, pickradius=6)
axs[1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
axs[1].grid("both")
axs[1].legend()
axs[2].plot(time,ch3, label='capteur 31',picker=True, pickradius=6)
axs[2].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
axs[2].grid("both")
axs[2].legend()

axs[-1].set_xlabel('time (s)')
axs[0].title.set_text("click to pick")
fig.set_size_inches(14,8)
fig.canvas.mpl_connect('pick_event', onpick)
plt.suptitle(filepath,alpha=.2,size=5)
plt.show()
plt.close('all')



fig, ax1 = plt.subplots()

ax1.set_xlabel('Position (m)')
ax1.set_ylabel('Time (µs)',color='b')
ax1.plot(x_sort,1e+6*(indexes_sort-np.min(indexes_sort)),'b',marker="+",linestyle="-.",linewidth=.5,label='rupture')
ax1.tick_params(axis='y', labelcolor='b')
ax1.grid(which='both')

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
ax2.set_ylabel('Time (µs)',color='orange')
ax2.plot(x_sensor,(indexes-np.min(indexes))*1e+6,'orange',marker="^",linestyle="-.",linewidth=.5,label='acoustic')
ax2.tick_params(axis='y', labelcolor='orange')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()