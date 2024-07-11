from nptdms import TdmsFile
import matplotlib.pyplot as plt
import scipy
import numpy as np
import sys

path = "E:/2023-2024/2024-05-16/sensorAndChannelsCompparison/"
filename = 'Sampling02.tdms'

filepath = path+filename
tdms_file = TdmsFile.read(filepath)
group = tdms_file['input']

decimate = 1
fs = 10e+6/decimate # sampling frequency (Hz)

ch1 = group['Voie 2'][::decimate]
ch2 = group['voie 3'][::decimate]
ch3 = group['voie 1'][::decimate]

Npoint = len(ch1)
time = np.linspace(0,Npoint/fs,Npoint)

fig, axs = plt.subplots(3,1,sharex=True,sharey=True,constrained_layout = True)
axs[0].plot(ch1,'b',label='capteur 11')
axs[0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
axs[0].grid("both")
axs[0].legend()
axs[1].plot(ch2,'orange',label='capteur 21')
axs[1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
axs[1].grid("both")
axs[1].legend()
axs[2].plot(ch3,'g',label='capteur 31')
axs[2].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
axs[2].grid("both")
axs[2].legend()

axs[-1].set_xlabel('time (s)')
axs[0].title.set_text("Acoustic traces")
fig.set_size_inches(14,8)
plt.suptitle(filepath,alpha=.2,size=5)
plt.show()
# plt.savefig(path+"acoustic_signal_all_all"+".png")

# plt.plot(time,event1,label='ampli 11, Energy : {}'.format(energy1))
# plt.plot(time,event2,label='ampli 21, Energy : {}'.format(energy2))
# plt.plot(time,event3,label='ampli 31, Energy : {}'.format(energy3))
# plt.ylabel('Count')
# plt.xlabel('Time')
# plt.title('Same acoustic event with same sensor on ampli 11,21 and 31')
# plt.grid("both")
# plt.legend()
# plt.show()