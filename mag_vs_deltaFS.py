'''
Petit programme pour comparer le saut de force cisaillante aux émissions acoustique lors d'un evenement trigger.
'''
import matplotlib.pyplot as plt
import numpy as np

try :
    from Python_DAQ import *
except :
    from DAQ_Python.Python_DAQ import *

loc_folder="E:/2023-2024/2024-05-30_sm-acoustic/"

Delta_Fs = np.zeros(38)
E_ac = np.zeros(38)

for i in range(38):
    if i<9:
        loc_file = "event-00"+str(i+1)+".npy"
    else: loc_file = "event-0"+str(i+1)+".npy"
    fsampl = 4e+6 # Hz
    loc=loc_folder+loc_file


    data=np.load(loc,allow_pickle=True)

    data=smooth(data,5)
    # data=np.transpose(np.transpose(data)-np.mean(data_zero,axis=1))

    fn_ev = voltage_to_force(data[15,:])
    fs_ev = voltage_to_force(data[31,:])
    trigger_ev = data[47,:]/1000
    ac_ev = data[63,:]

    fc = 100000
    Wn= 2*fc/fsampl
    N = 10 # order of the filter
    b, a = scipy.signal.butter(N, Wn, 'low')
    ac_ev_filt = scipy.signal.filtfilt(b, a, ac_ev)

    smoothing = np.where(np.abs(np.arange(-39995//2,39995//2,1))<500,1/500,0)
    fs_ev_filt = np.convolve(smoothing,fs_ev,mode='same')

    fs_moy = np.concatenate((np.ones(39995//2)*np.mean(fs_ev[:19997-100]),np.ones(39995//2+1)*np.mean(fs_ev[19997+100:])))

    E_ac[i] = np.sum(ac_ev**2)
    Delta_Fs[i] = fs_moy[-1]-fs_moy[0]

#     time = np.linspace(0,39995/fsampl,39995)
#     fig, ax1 = plt.subplots()
#     ax1.set_xlabel('Temps (s)')
#     ax1.set_ylabel('Count',color='blue')
#     #ax1.plot(time,trigger_ev,'blue',label='Trig')
#     ax1.plot(time,ac_ev_filt,'orange',label='Acoustique')
#     ax1.tick_params(axis='y', labelcolor='blue')
#     ax1.grid(which='both')
#     ax2 = ax1.twinx()
#     ax2.set_ylabel('Force (Kg)',color='green')
#     ax2.plot(time,fs_ev,'green',label='Fs',linestyle="-",linewidth=.25)
#     ax2.plot(time,fs_moy,'green',linestyle="--",label='Fs_moy')
#     ax2.tick_params(axis='y', labelcolor='green')
#     fig.tight_layout()
# plt.show()