'''
Petit programme obsolete de detection des evenements dans les fichiers tdms. Plutot utiliser dev_detection_event_multithreading.
'''
from nptdms import TdmsFile
import matplotlib.pyplot as plt
import scipy
import numpy as np
import sys

path = "E:/JulesLeBot2024/StageJules2024/Data/2024-06-25_full_manip/01/"
filename = 'Sampling01.tdms'
filepath = path+filename
tdms_file = TdmsFile.read(filepath)
try: group = tdms_file['input']
except (KeyError):
    print(tdms_file.groups())
    group = tdms_file['Analog inputs']
x_sensor = [0,0.2,0.35]

try :
    ch1,ch2,ch3 = group['Voie 2'][:],group['voie 3'][:],group['voie 1'][:]
except (KeyError):
    ch1,ch2,ch3 = group['Voie 2'][:],group['Voie 3'][:],group['Voie 1'][:]

index = []
for n in range((len(ch1)//1000000)-1):
    if np.max(ch1[n*1000000:(n+1)*1000000]) > 200:
        index.append([n*1000000,(n+1)*1000000])
if np.max(ch1[(n+1)*1000000:])> 200:
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

ch1 = ch1[Nstart:Nend]
ch2 = ch2[Nstart:Nend]
ch3 = ch3[Nstart:Nend]

fs = 5e+6 # sampling frequency (Hz)
time = np.linspace(0,Npoint/fs,Npoint)

ch1 = ch1/max(ch1)
ch2 = ch2/max(ch2)
ch3 = ch3/max(ch3)

f1, Pxx_den = scipy.signal.periodogram(ch1,fs)

f2, t, Sxx = scipy.signal.spectrogram(ch1,fs)

plt.figure(1)
plt.plot(time,ch1-2,label='ch1')
plt.plot(time,ch2,label='ch2')
plt.plot(time,ch3+2,label='ch3')
plt.xlabel('Time [s]')
plt.ylabel('Count')
plt.legend(loc='best')
plt.savefig(path+"acoustic_signal_all_"+str(ind)+".png")

plt.figure(2)
plt.loglog(f1, Pxx_den)
plt.xlabel('frequency [s^-1]')
plt.ylabel('PSD [?]')
plt.savefig(path+"periodogram_ch1_"+str(ind)+".png")


plt.figure(3)
plt.pcolormesh(t, f2,np.log(Sxx), shading='gouraud',vmin=-25)
plt.ylabel('Frequency [s^-1]')
plt.xlabel('Time [s]')
plt.ylim((0,1e+6))
plt.colorbar( label='ln(Sxx)')
plt.savefig(path+"spectrogram_ch1_"+str(ind)+".png")

plt.show()

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

slop = (x_sensor[1]-x_sensor[0])/(indexes[1]-indexes[0])
t = np.linspace(0,indexes[-1]-indexes[0],1000)

plt.scatter((indexes-indexes[0])*1e+6,x_sensor,marker="^")
plt.plot(t*1e+6,slop*t,linestyle="-.",linewidth=.25,color='black',label=r'propagation for C={:.2f}m/s'.format(slop))
plt.ylabel('Position (m)')
plt.xlabel('Time (µs)')
plt.title('Acoustic waves first arrival')
plt.grid('both')
plt.legend()
plt.savefig(path+"acoustic_waves_propagation_"+str(ind)+".png")
plt.show()