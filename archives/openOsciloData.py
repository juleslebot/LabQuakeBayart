import numpy as np
import matplotlib.pyplot as plt

path = 'E:/2023-2024/2024-05-22/'
file1 = path+'ALL0001/F0001CH1bis.CSV' # capteur (+ampli à cote)
file2 = path+'ALL0001/F0001CH2bis.CSV' # sorti de l'ampli
file3 = path+'ALL0000/F0000CH1bis.CSV' # capteur seul
file4 = 'Sampling_240522_mine_ampliYohann.tdms' # carte d'acq

Tacq_oscilo = 1e-4 # s
fs_o = 1/Tacq_oscilo
fs_dacq = 1e+6 # Hz


data1 = np.loadtxt(file1,delimiter=',',usecols=(3,4))
data2 = np.loadtxt(file2,delimiter=',',usecols=(3,4))
data3 = np.loadtxt(file3,delimiter=',',usecols=(3,4))
ch1,data4,ch3 = open_data(file4,path,'cont')

time4 = np.linspace(0,len(data4)/fs_dacq,len(data4)) -1.099

plt.figure()
plt.plot(data2[:,0],data2[:,1]/np.max(data2[:,1]),label='oscilo')
plt.plot(time4,data4/np.max(data4),label='acq')
plt.legend()
plt.xlabel('time (s)')

# plt.figure()
# plt.plot(time4,data4,label='acq')
# plt.legend()
# plt.xlabel('time (s)')
plt.show()