## Imports
# Science
import matplotlib.pyplot as plt
import numpy as np

try :
    from Python_DAQ import *
except :
    from DAQ_Python.Python_DAQ import *



sampling_freq_in=1000
ylabels=["$F_n$ (kg)", "$F_f$ (kg)", "TTL", "chan 2","chan 5","chan 8","chan 11","chan 14"]

loc="D:/Users/Manips/Documents/DATA/FRICS/2022/2022-04-21/Daq_Fn_Epsilonxx_allow_pivot_rubber.txt"
data=np.loadtxt(loc)

#Convert to Newtons
data[0]*=9.81

for i in range(3,8):
    data[i]-=np.min(data[i])


idex=np.array([i for i in range(len(data[0])) if data[0][i]>1000])







ms=[]

for i in range(3,8):
    m,b=np.polyfit(data[0][idex],data[i][idex],1)
    ms.append(m)
    plt.scatter(data[0],0.0001*i+data[i],label="chan {}, m={}".format(3*i-7,m),alpha=.5,s=5)
    plt.plot(data[0][idex],0.0001*i+b+m*data[0][idex],alpha=.5,c='b')

exp_nu=0.4
exp_E=3.1e9
S=1.5e-3
plt.title("Average slope : {:.3e}, Expected {:.3e}".format(np.average(ms[1:]),np.abs((1-exp_nu**2)/(S*exp_E))))
plt.xlabel(r"$F_N$ (N)")
plt.ylabel(r"$\varepsilon_{yy}$ (m/m)")
plt.legend()
plt.show()

