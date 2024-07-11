## Imports
# Science
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy import signal

from scipy import interpolate
try :
    from Python_DAQ import *
except :
    from DAQ_Python.Python_DAQ import *


#"plt.rcParams.update({"text.usetex": True})
#from matplotlib import rc
#rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
## read bin

location='D:/Users/Manips/Documents/DATA/FRICS/2022/2022-04-12/bin/'
#location='D:/Users/Manips/Downloads/'
#location='F:/'
file = 'osc_1_3.bin'
location='D:/Users/Manips/Documents/DATA/FRICS/2022/2022-06-10/npy/'

file='osc_2_5_epsilon_time_xx_yy_xy.npy'



#files = sorted(os.listdir(location))
#file=files[5]
#print(file)

###

time,data_tot= open_data_bin(location+file)

fs = round(1/np.median(time[1:]-time[:-1]))

data_tot_filtered=np.copy(data_tot)

###

#time=time[::1000]
#data_tot=data_tot[:,::1000]

###

data_save=np.concatenate((time.reshape((1,len(time))),data_tot),axis=0)

np.savetxt(location+file+".2MHZ.txt",data_save.T)