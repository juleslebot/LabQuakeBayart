### Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from matplotlib import rc
from scipy.optimize import curve_fit
from matplotlib.widgets import Slider, Button
import os

try :
    from Python_DAQ import *
except :
    from DAQ_Python.Python_DAQ import *

try :
    from full_epsilon import *
except :
    from DAQ_Python.full_epsilon import *
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif"})


### Constants
C_r=1255
C_s=1345
C_d=2332
nu=0.335
E=5.651e9
constants={"C_r":C_r,"C_s":C_s,"C_d":C_d,"nu":nu,"E":E}

time = np.arange(-0.0001,0.0001,2e-8)

names=['v     ','y_pos ','t_0   ','Gamma ']

args=np.array([500,0.003,0,1])

### Compute results

xx,yy,xy = full_epsilon(time,args,PlaneStress=True,constants=constants)
v,y_pos,t_0,Gamma=args

### Plot
fig, axes = plt.subplots(1, 3,sharex=True,figsize=(10, 4))

line1,=axes[0].plot(time*1e3,xx*1e6,color="b",alpha=0.3)
line2,=axes[1].plot(time*1e3,yy*1e6,color="b",alpha=0.3)
line3,=axes[2].plot(time*1e3,xy*1e6,color="b",alpha=0.3)

axes[1].set_xlabel("time (ms)")
axes[0].set_ylabel(r'$\varepsilon_{xx}$ (µm/m)')
axes[1].set_ylabel(r'$\varepsilon_{yy}$ (µm/m)')
axes[2].set_ylabel(r'$\varepsilon_{xy}$ (µm/m)')



for ax in axes:
    ax.grid(which='both')

#axes[-1].legend()

plt.tight_layout()


plt.show()



fig= plt.figure(figsize=(3.5, 3))

plt.plot(time*1e3,-xx*1e6,color="r",label=r"$-\varepsilon_{xx}$",alpha=0.5)
plt.plot(time*1e3,yy*1e6,color="g",label=r"$\varepsilon_{yy}$",alpha=0.5)
plt.plot(time*1e3,xy*1e6,color="b",label=r"$\varepsilon_{xy}$",alpha=0.5)
plt.axvline(0,linestyle="--",color='k',label=r"Crack tip")
plt.xlabel("Time (ms)")
plt.ylabel('Strain (µm/m)')

plt.grid(which='both')
plt.legend()
#axes[-1].legend()

plt.tight_layout()


plt.savefig('test.pdf',dpi=600)

















