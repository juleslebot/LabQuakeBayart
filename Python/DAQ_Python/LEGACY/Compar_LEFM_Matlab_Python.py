
import matplotlib.pyplot as plt
import numpy as np

### Constants

C_r=1255
C_s=1345
C_d=2332
nu=0.335
E=5.651e9
constants={"C_r":C_r,"C_s":C_s,"C_d":C_d,"nu":nu,"E":E}




try :
    from Python_DAQ import *
except :
    from DAQ_Python.Python_DAQ import *


try :
    from full_epsilon import *
except :
    from DAQ_Python.full_epsilon import *






dic=import_struc_from_matlab("D:/Users/Manips/Documents/DATA/FRICS/Data_Jeru_ForYohann/simul_lefm_1.mat")

time=dic['t']

v,y_pos,t_0,Gamma=985.6574,0.003,0,1

xx,yy,xy=full_epsilon(time,[v,y_pos,t_0,Gamma],PlaneStress=True,constants=None)


fig, axes = plt.subplots(1, 3,sharex=True)

line1,=axes[0].plot(time,xx,color="b",alpha=0.3)
line2,=axes[1].plot(time,yy,color="b",alpha=0.3)
line3,=axes[2].plot(time,xy,label="Python Simul",color="b",alpha=0.3)

line4,=axes[0].plot(time,dic["Uxx"],color="orange",alpha=0.3)
line5,=axes[1].plot(time,dic["Uyy"],color="orange",alpha=0.3)
line6,=axes[2].plot(time,dic["Uxy"],label="Matlab Simul",color="orange",alpha=0.3)

axes[1].set_xlabel("time (s)")
axes[0].set_ylabel(r'$\varepsilon_{xx}$')
axes[1].set_ylabel(r'$\varepsilon_{yy}$')
axes[2].set_ylabel(r'$\varepsilon_{xy}$')

for ax in axes:
    ax.grid(which="both")

plt.show()
