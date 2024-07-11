from nptdms import TdmsFile
import matplotlib.pyplot as plt
import scipy
import numpy as np
import sys

x = np.arange(0,100,1)
np.where(np.abs(x)<1,1,0)


def prepareFig(filename,path):
    filepath = path+filename
    tdms_file = TdmsFile.read(filepath)
    group = tdms_file['Analog inputs']
    ch1 = group['Voie 1'][:]

    index = []
    for n in range((len(ch1)//1000000)-1):
        if np.max(ch1[n*1000000:(n+1)*1000000]) > 200:
            index.append([n*1000000,(n+1)*1000000])
    if np.max(ch1[(n+1)*1000000:])> 200:
            index.append([(n+1)*1000000,(n+2)*1000000])

    for i,ind in enumerate(index):
        print(i,' : ',ind)

    ind = int(input('Choose index : '))
    Nstart = index[ind][0]
    Nend = index[ind][1]
    ch1 = group['Voie 2'][Nstart:Nend]
    Nstart = np.abs(ch1).argmax() - 200
    Nend = Nstart + 1200
    Npoint = Nend - Nstart

    filter = np.where(np.abs(np.arange(-Npoint//2,Npoint//2,1))<1,1/1,0)

    ch1 = np.convolve(filter,ch1[Nstart:Nend],mode='same')

    time = np.linspace(0,Npoint/fs,Npoint)
    plt.figure(1)
    energy = np.sum(ch1**2)
    ch1 /= np.sqrt(energy)
    energy_prime = np.sum(ch1**2)
    if j==0: plt.plot(time,ch1,label='file {}, Energy : {}'.format(j+1,energy))
    else: plt.plot(time,ch1,label='file {}, Energy : {}'.format(j+1,energy))
def show():
    plt.figure(1)
    plt.ylabel('Count')
    plt.grid("both")
    plt.title('ch1')
    plt.legend()
    plt.suptitle('Acoustic events')

    plt.show()

##=====================================================

fs = 10e+6 # sampling frequency (Hz)

indexes = np.zeros((3))

path = "E:/2023-2024/2024-05-15-acoustic-sensor-callibration/"
file = 'Sampling22-mine-d2-3coups-93mm.tdms'
energy = []

j=0
prepareFig(file,path)
j=2
prepareFig(file,path)
show()