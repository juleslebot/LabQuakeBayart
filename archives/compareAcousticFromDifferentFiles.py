'''
Petit programme pour plotter ensemble les données acoustiques depuis plusieurs fichiers.
Permet de controler la qualite du systeme d'acquisition.
'''
from nptdms import TdmsFile
import matplotlib.pyplot as plt
import scipy
import numpy as np
import sys


def prepareFig(filename,path):
    filepath = path+filename
    tdms_file = TdmsFile.read(filepath)
    group = tdms_file['Analog inputs']

    ch1 = group['Voie 2'][:]
    ch2 = group['Voie 3'][:]
    ch3 = group['Voie 1'][:]

    Nstart = ch1.argmax() - 30
    Nend = Nstart + 230
    Npoint = Nend - Nstart

    ch1 = group['Voie 2'][Nstart:Nend]
    ch2 = group['Voie 3'][Nstart:Nend]
    ch3 = group['Voie 1'][Nstart:Nend]
    time = np.linspace(0,Npoint/fs,Npoint)

    axs[0].plot(time,ch1,label='file '+str(j))
    axs[1].plot(time,ch2,label='file '+str(j))
    axs[2].plot(time,ch3,label='file '+str(j))
    energy.append([np.sum(ch1**2),np.sum(ch2**2), np.sum(ch3**2)])
    return energy

def show(energy):
    fig.set_size_inches(14,8)
    for i in (0,1,2):
        axs[i].set_ylabel('Count')
        axs[i].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        axs[i].grid("both")
        axs[i].set_title('ch'+str(i+1))
        axs[i].legend()
    plt.suptitle('Acoustic events')
    plt.savefig(path+"events_compared.png")

    plt.figure()
    energy = np.array(energy)
    plt.plot(energy[:,0],label='ch1',color='hotpink')
    plt.plot(energy[:,1],label='ch2',color='steelblue')
    plt.plot(energy[:,2],label='ch3',color='goldenrod')
    plt.ylabel('Energy')
    plt.xlabel('Index of the file')
    plt.legend()
    plt.grid()

    plt.show()

#=====================================================

fs = 10e+6 # sampling frequency (Hz)

fig, axs = plt.subplots(3,1,sharex=True,sharey=True,constrained_layout = True)
indexes = np.zeros((3))

path = "E:/2023-2024/2024-05-15-acoustic-sensor-callibration/"
energy = []
files = ['Sampling24-mine-d2-3coups-93mm.tdms','Sampling23-mine-d2-2coups-93mm.tdms','Sampling22-mine-d2-3coups-93mm.tdms']

for j,f in enumerate(files):
    energy = prepareFig(f,path)
show(energy)
