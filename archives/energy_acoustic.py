'''
Test sur la normalisation des signaux acoustiques par l'energie.
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
group = tdms_file['Analog inputs']

ch1 = group['Voie 2'][:]
ch2 = group['Voie 3'][:]
ch3 = group['Voie 1'][:]

Npoint = len(ch1)

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
axs[2].set_xlabel('index')
plt.show()

Nstart = int(input('Nstart ? '))
Nend = int(input('Nend ? '))

ch1 = ch1[Nstart:Nend]
ch2 = ch2[Nstart:Nend]
ch3 = ch3[Nstart:Nend]

energy = [np.sum(ch1**2),np.sum(ch2**2), np.sum(ch3**2)]

ch1_n = (ch1/energy[0]**.5)
ch2_n = (ch2/energy[1]**.5)
ch3_n = (ch3/energy[2]**.5)

# ch1_n = ch1/np.max(ch1)
# ch2_n = ch2/np.max(ch2)
# ch3_n = ch3/np.max(ch3)

fig, axs = plt.subplots(3,1,sharex=True,sharey=True,constrained_layout = True)
axs[0].plot(ch1_n,'b',label='capteur 11')
axs[0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
axs[0].grid("both")
axs[0].legend()
axs[1].plot(ch2_n,'orange',label='capteur 21')
axs[1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
axs[1].grid("both")
axs[1].legend()
axs[2].plot(ch3_n,'g',label='capteur 31')
axs[2].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
axs[2].grid("both")
axs[2].legend()
axs[2].set_xlabel('index')
plt.show()