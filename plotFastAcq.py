'''
Utilise les donnees traiter et ranger en dictionnaire (openFastAcq) pour comparer des quantité entre evenements ou entre jauges.
'''
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from scipy import interpolate
from Python.DAQ_Python.Python_DAQ import *
import glob
from tqdm import tqdm

dict = np.load('C:/Users/Manips/Desktop/tmp/2024-06-25_full_manip/02/manip_dict.npy',allow_pickle=True).all() # .item()

##
time = np.arange(0,39990,1)/dict['metadata']['sampling_freq']
fn_direct = np.zeros((39990))
fn_gages = np.zeros((39990))
fs_direct = np.zeros((39990))
fs_gages = np.zeros((39990))
nrj_sum = np.zeros((17,20))
nrj_max = np.zeros((17,20))
shear_drop = np.zeros((17,20))
for i in range(17):
    key1 = list(dict.keys())[2+i]
    fn_direct[i] = np.mean(dict[key1]['fn'])
    fn_gages[i] = dict[key1]['fn_gages']*1e-6
    fs_direct[i] = np.mean(dict[key1]['fs'])
    fs_gages[i] = dict[key1]['fs_gages']*1e-6
    for j in range(20):
        key2 = list(dict[key1].keys())[8+i]
        nrj_sum[i,j] = dict[key1][key2]['nrj_sum_signal']
        nrj_max[i,j] = dict[key1][key2]['nrj_max_stft']
        shear_drop[i,j] = dict[key1][key2]['delta_sigma_xy']


plt.plot(fn_gages,fn_direct,'.')
plt.xlabel('Fn')
plt.ylabel('Fn (kg)')
plt.grid()
plt.show()

plt.plot(fs_gages,fs_direct,'.')
plt.xlabel('Fn')
plt.ylabel('Fn (kg)')
plt.grid()
plt.show()

plt.plot(np.max(nrj_sum,axis=1),np.max(shear_drop,axis=1),'C0.')
plt.plot(np.max(nrj_max,axis=1),np.max(shear_drop,axis=1),'C1.')
plt.xlabel("NRJ")
plt.ylabel('Shear drop')
plt.grid()
plt.show()