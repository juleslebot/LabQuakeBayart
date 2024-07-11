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



### Constants
C_r=1255
C_s=1345
C_d=2332
#nu=0.335
#E=5.651e9




time_arr = np.arange(-0.0001,0.0001,5e-7)

### Fitting function

with open("./DAQ_Python/full_epsilon.py") as f:
    exec(f.read())
    f.close()

#set bounds
bounds=[
[ 10     ,C_r  ], #v
[ 1e-5 ,0.02 ], #y_pos
[min(time_arr),max(time_arr) ], #t_0
[ 1e-5    ,10     ],#Gamma
[0.01,1.99],#alpha
[0.001,0.02]]#thickness
bounds=np.array(bounds)
bounds=bounds.transpose()

# set proportion of the data to take into account

names=['v     ','y_pos ','t_0   ','Gamma ','alpha','e']



### Optimization computation

args=np.array([500,0.003,0,1,1,0.01])

### Compute results

xx,yy,xy = full_epsilon_alpha(time_arr,args,PlaneStress=False)
xxp,yyp,xyp = full_epsilon_alpha(time_arr,args,PlaneStress=True)


xx0,yy0,xy0 = full_epsilon(time_arr,args[:4],PlaneStress=False)
xxp0,yyp0,xyp0 = full_epsilon(time_arr,args[:4],PlaneStress=True)

v,y_pos,t_0,Gamma,alpha,e=args

### Plot
fig, axes = plt.subplots(1, 3,sharex=True)

line1,=axes[0].plot(1000*time_arr,xx,color="b",alpha=0.5)
line2,=axes[1].plot(1000*time_arr,yy,color="b",alpha=0.5)
line3,=axes[2].plot(1000*time_arr,xy,label="Plane Strain",color="b",alpha=0.5)

line4,=axes[0].plot(1000*time_arr,xxp,color="orange",alpha=0.5)
line5,=axes[1].plot(1000*time_arr,yyp,color="orange",alpha=0.5)
line6,=axes[2].plot(1000*time_arr,xyp,label="Plane Stress",color="orange",alpha=0.5)





line10,=axes[0].plot(1000*time_arr,xx0,color="b",alpha=0.5,linestyle=':')
line20,=axes[1].plot(1000*time_arr,yy0,color="b",alpha=0.5,linestyle=':')
line30,=axes[2].plot(1000*time_arr,xy0,label="1D equivalent",color="b",alpha=0.5,linestyle=':')

line40,=axes[0].plot(1000*time_arr,xxp0,color="orange",alpha=0.5,linestyle=':')
line50,=axes[1].plot(1000*time_arr,yyp0,color="orange",alpha=0.5,linestyle=':')
line60,=axes[2].plot(1000*time_arr,xyp0,color="orange",alpha=0.5,linestyle=':')







axes[1].set_xlabel("time (ms)")
axes[0].set_ylabel(r'$\varepsilon_{xx}$')
axes[1].set_ylabel(r'$\varepsilon_{yy}$')
axes[2].set_ylabel(r'$\varepsilon_{xy}$')




for ax in axes:
    ax.grid(which='both')

axes[-1].legend()

plt.tight_layout()
#plt.show()


plt.subplots_adjust(bottom=0.6)


axfreq1 = plt.axes([0.25, 0.1, 0.65, 0.03])
v_slider = Slider(
    ax=axfreq1,
    label='v [m/s]',
    valmin=bounds[0,0],
    valmax=bounds[1,0],
    valinit=v,
)

axfreq2 = plt.axes([0.25, 0.15, 0.65, 0.03])
y_pos_slider = Slider(
    ax=axfreq2,
    label='y_pos [m]',
    valmin=bounds[0,1],
    valmax=bounds[1,1],
    valinit=y_pos,
)

axfreq3 = plt.axes([0.25, 0.2, 0.65, 0.03])
t_0_slider = Slider(
    ax=axfreq3,
    label='t_0 [s]',
    valmin=bounds[0,2],
    valmax=bounds[1,2],
    valinit=t_0,
)

axfreq4 = plt.axes([0.25, 0.25, 0.65, 0.03])
Gamma_slider = Slider(
    ax=axfreq4,
    label='$\Gamma$ [J/m]',
    valmin=bounds[0,3],
    valmax=bounds[1,3],
    valinit=Gamma,
)


axfreq5 = plt.axes([0.25, 0.3, 0.65, 0.03])
alpha_slider = Slider(
    ax=axfreq5,
    label=r'$\alpha/(\pi/2)$',
    valmin=bounds[0,4],
    valmax=bounds[1,4],
    valinit=alpha,
)

axfreq6 = plt.axes([0.25, 0.35, 0.65, 0.03])
e_slider = Slider(
    ax=axfreq6,
    label=r'$e$ [m]',
    valmin=bounds[0,5],
    valmax=bounds[1,5],
    valinit=e,
)

def update(val):
    args_temp=(v_slider.val, y_pos_slider.val,t_0_slider.val,Gamma_slider.val,alpha_slider.val,e_slider.val)
    xx,yy,xy = full_epsilon_alpha(time_arr,args_temp, PlaneStress=False)
    xxp,yyp,xyp = full_epsilon_alpha(time_arr,args_temp, PlaneStress=True)


    xx0,yy0,xy0 = full_epsilon(time_arr,args_temp[:4],PlaneStress=False)
    xxp0,yyp0,xyp0 = full_epsilon(time_arr,args_temp[:4],PlaneStress=True)


    line1.set_ydata(xx)
    line2.set_ydata(yy)
    line3.set_ydata(xy)
    line4.set_ydata(xxp)
    line5.set_ydata(yyp)
    line6.set_ydata(xyp)

    line10.set_ydata(xx0)
    line20.set_ydata(yy0)
    line30.set_ydata(xy0)
    line40.set_ydata(xxp0)
    line50.set_ydata(yyp0)
    line60.set_ydata(xyp0)

    fig.canvas.draw_idle()



v_slider.on_changed(update)
y_pos_slider.on_changed(update)
t_0_slider.on_changed(update)
Gamma_slider.on_changed(update)
alpha_slider.on_changed(update)
e_slider.on_changed(update)

#plt.xlim([-0.1,0.05])
plt.subplots_adjust(wspace=0.1)


plt.show()














