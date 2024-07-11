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



### Constants
C_r=1255
C_s=1345
C_d=2332
nu=0.335
E=4e9
constants={"C_r":C_r,"C_s":C_s,"C_d":C_d,"nu":nu,"E":E}

time = np.arange(-0.005,0.005,2e-8)

### Fitting function


#set bounds
bounds=[
[ 10     ,C_r  ], #v
[ 1e-5 ,0.02 ], #y_pos
[min(time),max(time) ], #t_0
[ 1e-5    ,10     ]] #Gamma
bounds=np.array(bounds)
bounds=bounds.transpose()

# set proportion of the data to take into account

names=['v     ','y_pos ','t_0   ','Gamma ']



### Optimization computation

args=np.array([500,0.003,0,1])

### Compute results

xx,yy,xy = full_epsilon(time,args,PlaneStress=False,constants=constants)
xxp,yyp,xyp = full_epsilon(time,args,PlaneStress=True,constants=constants)
v,y_pos,t_0,Gamma=args

### Plot
fig, axes = plt.subplots(1, 3,sharex=True)

line1,=axes[0].plot(time,xx,color="b",alpha=0.3)
line2,=axes[1].plot(time,yy,color="b",alpha=0.3)
line3,=axes[2].plot(time,xy,label="Plane Strain",color="b",alpha=0.3)

line4,=axes[0].plot(time,xxp,color="orange",alpha=0.3)
line5,=axes[1].plot(time,yyp,color="orange",alpha=0.3)
line6,=axes[2].plot(time,xyp,label="Plane Stress",color="orange",alpha=0.3)

axes[1].set_xlabel("time (s)")
axes[0].set_ylabel(r'$\varepsilon_{xx}$')
axes[1].set_ylabel(r'$\varepsilon_{yy}$')
axes[2].set_ylabel(r'$\varepsilon_{xy}$')



for ax in axes:
    ax.grid(which='both')

axes[-1].legend()

plt.tight_layout()
#plt.show()


plt.subplots_adjust(bottom=0.6)


axfreq1 = plt.axes([0.25, 0.2, 0.65, 0.03])
v_slider = Slider(
    ax=axfreq1,
    label='v [m/s]',
    valmin=bounds[0,0],
    valmax=bounds[1,0],
    valinit=v,
)

axfreq2 = plt.axes([0.25, 0.3, 0.65, 0.03])
y_pos_slider = Slider(
    ax=axfreq2,
    label='y_pos [m]',
    valmin=bounds[0,1],
    valmax=bounds[1,1],
    valinit=y_pos,
)

axfreq3 = plt.axes([0.25, 0.4, 0.65, 0.03])
t_0_slider = Slider(
    ax=axfreq3,
    label='t_0 [s]',
    valmin=bounds[0,2],
    valmax=bounds[1,2],
    valinit=t_0,
)

axfreq4 = plt.axes([0.25, 0.5, 0.65, 0.03])
Gamma_slider = Slider(
    ax=axfreq4,
    label='$\Gamma$ [J/m]',
    valmin=bounds[0,3],
    valmax=bounds[1,3],
    valinit=Gamma,
)

def update(val):
    xx,yy,xy = full_epsilon(time,(v_slider.val, y_pos_slider.val,t_0_slider.val,Gamma_slider.val), PlaneStress=False,constants=constants)
    xxp,yyp,xyp = full_epsilon(time,(v_slider.val, y_pos_slider.val,t_0_slider.val,Gamma_slider.val), PlaneStress=True,constants=constants)
    line1.set_ydata(xx)
    line2.set_ydata(yy)
    line3.set_ydata(xy)
    line4.set_ydata(xxp)
    line5.set_ydata(yyp)
    line6.set_ydata(xyp)
    fig.canvas.draw_idle()



v_slider.on_changed(update)
y_pos_slider.on_changed(update)
t_0_slider.on_changed(update)
Gamma_slider.on_changed(update)

#plt.xlim([-0.1,0.05])
plt.subplots_adjust(wspace=0.1)

plt.show()














