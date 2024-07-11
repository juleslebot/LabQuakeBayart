import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from scipy import interpolate
import glob
from tqdm import tqdm

import Python.DAQ_Python.Python_DAQ as daq
import Python.DAQ_Python.bib_fastacq_jules as facq
## Load

loc='C:/Users/Manips/Desktop/StageJules2024/Data/2024-06-25_full_manip/01/'
file_zero = "event-001.npy"
file_sm = "slowmon.npy"
params="parameters.txt"

data_zero = np.load(loc+file_zero,allow_pickle=True)
data_sm = np.load(loc+file_sm,allow_pickle=True)
time_sm = np.load(loc+"slowmon_time.npy")

files = 'all'
if files == 'all':
    files = glob.glob(loc+"event-0**.npy")

## Create dict
gages_ch = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62])
acq = {}

acq['metadata'] = { 'sampling_freq' : 1e6, # Hz
                    'gages_nb' : 60,
                    'gages_pos' : np.linspace(0,0.1425,20),
                    'h_grains': 6e-3, # m
                    'path_to_camera' : 'None',
                    'path_to_acoustic' : 'C:/Users/Manips/Desktop/tmp/2024-06-25_full_manip/04_nocamera/Sampling04.tdms',
                    'desciption':'Manip avec grains à l interface, camera, 60 jauges (donc 20 rosettes), 1/2 rosette de chaque cote du bloc, nombre pair du cote camera, acquisition acoustique a 1MHz'
                    }

## SM
acq['sm'] = {  'fn' : daq.voltage_to_force(data_sm[15,:]),
        'fs' : daq.voltage_to_force(data_sm[31,:]),
        'trig' : data_sm[47,:],
        'acou' : data_sm[63,:],
        'gages' : data_sm[gages_ch],
        'time' : time_sm}
# conversion
for i in range(len(gages_ch)//3):
    ch_1=acq['sm']['gages'][3*i]
    ch_2=acq['sm']['gages'][3*i+1]
    ch_3=acq['sm']['gages'][3*i+2]
    ch_1,ch_2,ch_3=daq.voltage_to_strains(ch_1,ch_2,ch_3,amp=2000)
    ch_1,ch_2,ch_3=daq.rosette_to_tensor(ch_1,ch_2,ch_3)
    acq['sm']['gages'][3*i]=ch_1
    acq['sm']['gages'][3*i+1]=ch_2
    acq['sm']['gages'][3*i+2]=ch_3


acq['sm']['eps_yy'] = acq['sm']['gages'][1::3]
acq['sm']['eps_xy']= acq['sm']['gages'][2::3]
acq['sm']['eps_xx'] = acq['sm']['gages'][0::3]
del acq['sm']['gages']
acq['sm']['mu'] = daq.smooth(acq['sm']['fs'],50) / daq.smooth(acq['sm']['fn'],50)

### ACQ
time_fast_acq = np.arange(0,data_zero.shape[1],1)/acq['metadata']['sampling_freq']
rosette_pos = acq['metadata']['gages_pos']

def treat_event(file):
    data_acq = np.load(file,allow_pickle=True)
    data_acq = daq.smooth(data_acq,10) # smoothing data
    data_acq = np.transpose(np.transpose(data_acq)-np.mean(data_zero,axis=1)) # callibrate data set

    fs = daq.voltage_to_force(data_acq[31,:])
    fn = daq.voltage_to_force(data_acq[15,:])
    acoustic = data_acq[63,:]

    eps_xx,eps_xy,eps_yy = facq.signal_to_tensor(data_acq[gages_ch])
    mu = fs.mean() / fn.mean()
    delta_sigma_xy_moy, delta_sigma_xy_click, c_rupture, sigma_yy_dx, sigma_xy_dx = facq.detect_stress_drop(eps_xx,eps_xy,eps_yy,rosette_pos)
    nrj_sum_signal = facq.nrj(data_acq[gages_ch],'sum')
    nrj_max_stft = facq.nrj(data_acq[gages_ch],'max')

    event_dict = {
            'fn' : fn,
            'fs' : fs,
            'trig' : data_acq[47,:],
            'acoustic' : acoustic,
            'mu' : mu,
            'rupture_velocity': c_rupture,
            'fn_gages':sigma_yy_dx,
            'fs_gages': sigma_xy_dx}

    for i in range(len(rosette_pos)):
        event_dict['rosette{}'.format(i+1)] = {
            'eps_xx' : eps_xx[i,:],
            'eps_xy' : eps_xy[i,:],
            'eps_yy' : eps_yy[i,:],
            'delta_sigma_xy_moy': delta_sigma_xy_moy[i],
            'delta_sigma_xy_click': delta_sigma_xy_click[i],
            'nrj_sum_signal' : nrj_sum_signal[i],
            'nrj_max_stft' : nrj_max_stft[i]}

    return event_dict

for file in tqdm(files):
    key = file[-13:-4]
    acq[key] = treat_event(file)

# il faudrait reorganiser le tout, on voudrait avoir
# Manips/
#   -fs
#   -fn
#   -trig
#   -acoustic_sm
#   -Events/
#       - fast_time
#       - intergrale des conraintes
#       - vitesse de la rupture
#       -01/
#           -rosette 01/
#               -toute la fast acq (fn,fs,eps_xx,xy,yy)
#               -tout les traitements (mu, delta_sigma, nrj...)
#            -...
#            -rosette 20/
#       -...
#       -82/ (ou autre)

## Save
np.save(loc+'manip_dict',acq)