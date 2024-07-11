'''
Premiere tentative de regrouper des fonctions utiles au traitement des donnees acoustique.
Plutot utiliser dev_stats_detection...
'''
from nptdms import TdmsFile
import matplotlib.pyplot as plt
import scipy
import numpy as np
import sys

# open data from files

def open_data(filepath,mode):
    tdms_file = TdmsFile.read(filepath)
    if mode == 'trig':
        group = tdms_file['input']
        ch1 = group['Voie 2'][:]
        ch2 = group['voie 3'][:]
        ch3 = group['voie 1'][:]
    if mode == 'cont':
        group = tdms_file['Analog inputs']
        ch1 = group['Voie 2'][:]
        ch2 = group['Voie 3'][:]
        ch3 = group['Voie 1'][:]
    return ch1,ch2,ch3

# choose the part of data we want to work on

def look_for_event(data,Nsection,threshold):
    index = []
    for n in range((len(data)//Nsection)-1):
        if np.max(ch1[n*Nsection:(n+1)*Nsection]) > threshold:
            index.append([n*Nsection,(n+1)*Nsection])
    if np.max(ch1[(n+1)*Nsection:])> threshold:
            index.append([(n+1)*Nsection,(n+2)*Nsection])
    return index

def choose_event(data,Nbefore,Nafter):
    index = look_for_event(data,Nection=len(data//100),threshold=200)
    for i,ind in enumerate(index):
        print(i,' : ',ind)
    ind = int(input('Choose index : '))
    Nstart = index[ind][0]
    Nend = index[ind][1]
    data = data[Nstart:Nend]
    return data

def index_event(data,Nbefore,Nafter):
    Nstart = np.abs(data).argmax() - Nbefore
    Nend = Nstart + Nbefore +  Nafter
    return Nstart,Nend

# process data

def smooth_data(data,Nsmooth):
    Npoint = len(data)
    smoothing = np.where(np.abs(np.arange(-Npoint//2,Npoint//2,1))<Nsmooth,1/Nsmooth,0)
    smoothed_data = np.convolve(smoothing,data,mode='same')
    return smoothed_data

def compute_energy(data):
    return np.sum(data**2)

def normalize_data(data,reference_event):
    return data / np.sum(reference_event**2)**.5

def onpick(event):
    x_coord = event.mouseevent.xdata
    y_coord = event.mouseevent.ydata
    ax=event.mouseevent.inaxes
    col=(ax.get_subplotspec().rowspan.start)
    print(r'Picked point: ({:.2f}, {:.2f})'.format(1000*x_coord,1000*y_coord))
    indexes[col]=x_coord
def pick_acoustic_arrival(ch1,ch2,ch3):
    fig, axs = plt.subplots(3,1,sharex=True,sharey=True,constrained_layout = True)
    indexes = np.zeros((3))
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
    return indexes

def magnitude_distribution(ch,Nsection=10000,threshold=50,Nbefore=200,Nafter=1500):
    index = look_for_event(ch,Nsection,threshold)
    Nevent = len(index)
    event = np.zeros((Nevent,1700))
    energy = np.zeros(Nevent)
    for i,ind in enumerate(index):
        data = ch[ind[0]:ind[1]]
        Nstart, Nend = index_event(data,Nbefore,Nafter)
        if Nstart <0:
            event[i,:] = np.concatenate((ch[ind[0]+Nstart:ind[0]],data[0:Nend]))
        elif Nend > 10000:
            event[i,:] = np.concatenate((data[Nstart:],ch[ind[1]:ind[0]+Nend]))
        else :
            event[i,:] = data[Nstart:Nend]
        energy[i] = np.sum(event[i,:]**2)
    energy = energy/energy[0]
    return energy


# plot data

def plot_data_3channel(filepath,ch1,ch2,ch3,fs,decimate=1):
    Npoint = len(ch1[::decimate])
    fs = fs/decimate
    time = np.linspace(0,Npoint/fs,Npoint)
    fig, axs = plt.subplots(3,1,sharex=True,sharey=True,constrained_layout = True)
    axs[0].plot(time,ch1[::decimate],'b',label='capteur 11')
    axs[0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axs[0].grid("both")
    axs[0].legend()
    axs[1].plot(time,ch2[::decimate],'orange',label='capteur 21')
    axs[1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axs[1].grid("both")
    axs[1].legend()
    axs[2].plot(time,ch3[::decimate],'g',label='capteur 31')
    axs[2].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axs[2].grid("both")
    axs[2].legend()
    axs[-1].set_xlabel('Time (s)')
    axs[0].title.set_text("Acoustic traces")
    fig.set_size_inches(14,8)
    plt.suptitle(filepath,alpha=.2,size=5)


# routine

def routine1_plot_all(path,filename,type='cont',fs=1e+6):
    ch1,ch2,ch3 = open_data(path+filename,type)
    plot_data_3channel(path+filename,ch1,ch2,ch3,fs)
    plt.savefig(path+'all_acoustic.png')
    plt.show()
