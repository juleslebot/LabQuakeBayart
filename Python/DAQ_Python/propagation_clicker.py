## Imports
# Science
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from scipy import interpolate


#
# def rosette_to_tensor(ch_1,ch_2,ch_3):
#     """
#     converts a 45 degres rosette signal into a full tensor.
#     input : the three channels of the rosette
#     output : $\epsilon_{xx},\epsilon_{yy},\epsilon_{xy}
#     https://www.efunda.com/formulae/solid_mechanics/mat_mechanics/strain_gage_rosette.cfm
#     """
#     eps_xx=ch_1+ch_3-ch_2
#     eps_yy=ch_2
#     eps_xy=(ch_1-ch_3)/2
#     return(eps_xx,eps_yy,eps_xy)


try :
    from Python_DAQ import *
except :
    from DAQ_Python.Python_DAQ import *

### load data

loc_folder="D:/Users/Manips/Documents/DATA/FRICS/2023/2023-06-27-granular-light-matlab-export/manip_4/"
loc_file="event-005.npy"
loc_file_zero = "event-001.npy"
loc_params="parameters.txt"

# just in case there's nothing in the saved params
x=np.array([0,2.5,4.5,6.5,8.5])

exec(load_params(loc_folder+loc_params))


loc=loc_folder+loc_file
loc_zero=loc_folder+loc_file_zero
sampling_freq_in = clock/10

# Setup
speedup=1
speedup_smooth=1
roll_smooth=51
start=0
navg=1


data=np.load(loc,allow_pickle=True)
data_zero=np.load(loc_zero,allow_pickle=True)

interest = np.arange(0,15)

ylabels=[r"$\varepsilon_{{xx}}^{{{}}}$",r"$\varepsilon_{{yy}}^{{{}}}$",r"$\varepsilon_{{xy}}^{{{}}}$"]*5
for i in range(15):
    ylabels[i]=ylabels[i].format((i+3)//3)

forces=data[(16,17),:]
mu=data[17,:100].mean()/data[16,:100].mean()
data=data[interest]
data_zero=data_zero[interest]


### Convert voltage to strains


def voltage_to_strains(ch1,ch2,ch3):
    a=lambda x: V_to_strain(x,amp=495,G=1.79,i_0=0.0017,R=350)
    b=lambda x: -V_to_strain(x,amp=495,G=1.86,i_0=0.0017,R=350)
    ch1=a(ch1)
    ch2=b(ch2)
    ch3=a(ch3)
    return(ch1,ch2,ch3)




for i in range(5):
    ch_1=data[3*i]
    ch_2=data[3*i+1]
    ch_3=data[3*i+2]
    ch_1,ch_2,ch_3=voltage_to_strains(ch_1,ch_2,ch_3)
    a,b,c=rosette_to_tensor(ch_1,ch_2,ch_3)
    data[3*i]=a
    data[3*i+1]=b
    data[3*i+2]=c

for i in range(5):
    ch_1=data_zero[3*i]
    ch_2=data_zero[3*i+1]
    ch_3=data_zero[3*i+2]
    ch_1,ch_2,ch_3=voltage_to_strains(ch_1,ch_2,ch_3)
    a,b,c=rosette_to_tensor(ch_1,ch_2,ch_3)
    data_zero[3*i]=a
    data_zero[3*i+1]=b
    data_zero[3*i+2]=c



### smooth data




data_smooth=smooth(data,roll_smooth)
data_smooth=np.array([avg_bin(data_smooth[i],speedup_smooth) for i in range(len(data_smooth))])

forces_smooth=smooth(forces,roll_smooth)
forces_smooth=np.array([avg_bin(forces_smooth[i],speedup_smooth) for i in range(len(forces_smooth))])

#data=np.transpose(np.transpose(data)-data[:,start])
data_smooth=np.transpose(np.transpose(data_smooth)-np.mean(data_zero,axis=1))



# !!! WEIRD CHOICE, DON'T RUN THIS SECTION ALONE
data=data_smooth
forces=forces_smooth


### Plot macroscopic checks
sm=np.load(loc_folder+"Full_Daq.npy",allow_pickle=True).all()
time_sm=np.load(loc_folder+"slowmon_time.npy")
eps_yy=-sm["data"][[1,4,7,10,13]]
fn=-sm["data"][15]
fs=-sm["data"][16]
sm=np.load(loc_folder+"slowmon.npy",allow_pickle=True)
trigger=100*sm[18]
plt.plot(time_sm,fn)
plt.plot(time_sm,fs)
plt.plot(time_sm,trigger)
plt.xlabel("time (s)")
plt.title("Fn, Fs and trigger")
plt.grid()
plt.savefig(loc_folder+"slowmon_forces.png")
plt.show()
plt.close()
colors=["r","orange","gold","green","blue"]

for i in range(5):
    plt.plot(smooth(time_sm,100),smooth(eps_yy[i]-eps_yy[i][0:100].mean(),100),label="x={} m".format(x[i]), alpha=.7,color=colors[i])

plt.grid(which="both")
plt.xlabel("time (s)")
plt.ylabel(r"$\varepsilon_{yy}(t)$")
plt.legend()
plt.tight_layout()
plt.savefig(loc_folder+"slowmon_eps_yy.png")
plt.show()
plt.close()

plt.plot(x,np.mean(data[1::3,:1000],axis=1))
plt.xlabel("position (m)")
plt.ylabel(r"$\varepsilon_{yy}$ (m/m)")
plt.ylim([0,1.1*np.max(np.mean(data[1::3,:1000],axis=1))])
plt.grid(which="both")
plt.savefig(loc_folder+"{}_load_profile.png".format(loc_file[:-4]))
plt.show()
plt.close()

plt.plot(x,np.mean(-data[2::3,:1000],axis=1)/np.mean(-data[1::3,:1000],axis=1))
plt.xlabel("position (m)")
plt.ylabel(r"$\varepsilon_{xy}/\varepsilon_{yy}$")
plt.grid(which="both")
plt.savefig(loc_folder+"{}_load_profile_mu.png".format(loc_file[:-4]))
plt.show()
plt.close()





### Pick Events


sampling_freq_in=sampling_freq_in/speedup_smooth

time=np.arange(len(data[0]))/sampling_freq_in*navg
plt.plot(time,-500*forces[1,:]/3)
plt.xlabel("time (s)")
plt.ylabel(r"$F_s$ (kg)")
plt.grid(which="both")
plt.show()


loc_temp = "./.indexes_temp.npy"

indexes = np.zeros((3,5))
np.save(loc_temp,indexes)


def onpick(event):
    indexes = np.load(loc_temp)
    x_coord = event.mouseevent.xdata
    y_coord = event.mouseevent.ydata
    ax=event.mouseevent.inaxes
    row=(ax.get_subplotspec().rowspan.start)
    col=(ax.get_subplotspec().colspan.start)
    print(f'Picked point: ({x_coord:.2f}, {y_coord:.2f})')
    indexes[row,col]=x_coord
    np.save(loc_temp,indexes)


fig, axs = plt.subplots(3,5,sharex=True,sharey=False)

for i in range(len(data)):
    axs[i%3][i//3].plot(time[start::speedup],data[i][start::speedup], label=ylabels[i], picker=True, pickradius=6)
    axs[i%3][i//3].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axs[i%3][i//3].grid("both")
    axs[i%3][i//3].legend()
    #axs[i%3][i//3].axvline(time[find_event(newdata[i])])

axs[-1][0].set_xlabel('time (s)')
fig.set_size_inches(14,8)
fig.canvas.mpl_connect('pick_event', onpick)

plt.show()

plt.close()

indexes = np.load(loc_temp)


### Show propagation as picked in \varepsilon_{xy}

E=2.78e9


fig, axs = plt.subplots(5,1,sharex=True,sharey=False)

for i in range(5):
    axs[i].plot(1000*time[start::speedup]-1000*time.mean(),(data[3*(i+1)-1][start::speedup]-data[3*(i+1)-1][0:10000].mean())*E*1e-6, label="$\Delta\sigma_{{xy}}^{{{}}}$, $x={{{}}}$".format(i+1,x[i]))
    axs[i].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axs[i].grid("both")
    #plt.ticklabel_format(style='plain', axis='y')
    #plt.yticks([0,-0.25])
    #plt.ylim([-0.35,0.15])
    axs[i].legend(loc='upper right')
    axs[i].axvline(1000*indexes[2][i]-1000*time.mean(),color='k',linestyle="dashed")
    axs[i].axhline(0,color='grey',linestyle="solid",alpha=.5,linewidth=2)

axs[-1].set_xlim([-0.15,0.05])
axs[-1].set_xlabel('time (ms)')
axs[2].set_ylabel('$\sigma_{xy}$ (MPa)')

plt.savefig(loc_folder+loc_file+"propagation_verticale.png")
plt.show()
plt.close()


### Print times and speeds (xy)
print(r"µ={:.3f}".format(mu))

times=indexes[-1,:]

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





x_err = np.ones(5)*1e-3

times=(indexes.transpose()-np.mean(indexes,axis=1)).transpose()


y=[times[2,i] for i in range(5)]
# y=[np.mean(times[:,i]) for i in range(5)]
#y_err=[np.std(times[:,i]) for i in range(5)]
y_err=[1e-5 for i in range(5)]


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
plt.scatter(x,times[2],label=r"$\varepsilon_{xy}$")
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
plt.savefig(loc+"_propagation.png",dpi=600)
plt.show()
plt.close()




