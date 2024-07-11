import numpy as np
import matplotlib.pyplot as plt
Tforce=np.array([68.22,74.52,81.19,87.13,93.48,99.32,103.97,108.83,113.58,116.79,120.72,126.50,132.29,137.46,142.83])
Tpos=np.array([65.5,71.9,84.3,90.9,96.5,101.3,106.1,110.5,114.3,117.9,123.7,129.1,134.9,139.9,143.7])
ntour=np.array([i for i in range(len(Tforce))])


Tforce=Tforce-Tforce[0]
Tpos=Tpos-Tpos[0]


coef = np.polyfit(ntour, Tforce, 1)
poly1d_fn = np.poly1d(coef)

coef2 = np.polyfit(ntour, Tpos, 1)
poly1d_fn2 = np.poly1d(coef2)

plt.plot(Tforce,ntour, label='Sauts de force',marker="+")
plt.plot(Tpos,ntour, label='Sauts de Position (caméra)',marker="+")
plt.plot(poly1d_fn(ntour),ntour, '--b',label="Regression (force)") #'--k'=black dashed line, 'yo' = yellow circle marker
plt.plot(poly1d_fn2(ntour), ntour, '--y',label="Regression (position)") #'--k'=black dashed line, 'yo' = yellow circle marker
plt.legend()

plt.xlabel("Temps")
plt.ylabel("Nombre de sauts")

plt.grid()
plt.show()

print("Estimated turns per second : ")
print(0.5*(1/coef2[0]+1/coef[0]))


#plt.plot(Tforce,Tpos)

###

loc = "D:/Users/Manips/Documents/DATA/FRICS/2022/2022-06-30/daq2.txt"


speedup=1
sampling_freq_in=1000

ylabels=["$F_n$ (kg)", "$F_f$ (kg)", "Accelerometer(A.U.)"]#"Trigger (V)"]
#ylabels=["$F_n$ (kg)", "$F_f$ (kg)", "TTL", "chan 2","chan 5","chan 8","chan 11","chan 14"]


data=np.loadtxt(loc)

time=np.arange(len(data[0]))/sampling_freq_in*navg
time=time+(117.7-120.6)

fig, axs = plt.subplots(len(data))




for i in range(len(data)):
    axs[i].plot(time[::speedup],data[i][::speedup], label=ylabels[i])
    #axs[i].grid("both")
    axs[i].legend()

axs[-1].set_xlabel('time (s)')
fig.suptitle(loc[31:])


index=[0,1,2,3,4]
drops=[6.7,17.9,29.3,39.9,50.9]

coef3 = np.polyfit(index, drops, 1)
poly1d_fn3 = np.poly1d(coef3)

for i in range(14) :
    axs[1].axvline(poly1d_fn3(i),linestyle='dashed',linewidth=0.5,color='k')
    #axs[1].axvline(poly1d_fn3(i+0.5),linestyle='dashed',linewidth=0.5,color='g')

plt.show()



###

index=np.array([0,1,2,3,4])
drops=[8.72,23.06,35.5,46.88,60.1]

coef3 = np.polyfit(index, drops, 1)
poly1d_fn3 = np.poly1d(coef3)


plt.plot(coef3[0]*index+coef3[1],linestyle='dashed',linewidth=0.5,color='k',label="intervale entre chaque saut : {}, soit {:.4f} Hz ".format(coef3[0],1/coef3[0]))
plt.scatter(index,drops)
    #axs[1].axvline(poly1d_fn3(i+0.5),linestyle='dashed',linewidth=0.5,color='g')
plt.legend()
plt.xlabel("index")
plt.ylabel("t (s)")
plt.xticks([0,1,2,3,4],[0,1,2,3,4])
plt.grid()
plt.show()


###
drops=np.array([0.2485,0.2515,0.2553,0.2587,0.2619,0.2651,0.2689,0.2723,0.2760,0.2794,0.2830,0.2868,0.29,0.2932,0.2970,0.3002,0.3034,0.3070,0.3108,0.3138,0.3170,0.3202,0.3244])
index=np.arange(len(drops))



coef3 = np.polyfit(index, drops, 1)
poly1d_fn3 = np.poly1d(coef3)


plt.plot(coef3[0]*index+coef3[1],linestyle='dashed',linewidth=0.5,color='k',label="intervale entre chaque saut : {:.3f}µm ".format(1000*coef3[0]))
plt.scatter(index,drops)
    #axs[1].axvline(poly1d_fn3(i+0.5),linestyle='dashed',linewidth=0.5,color='g')
plt.legend()
plt.xlabel("index")
plt.ylabel("$\delta x$ (mm)")
plt.grid()
plt.show()






###



coef4=np.polyfit(time,data[3],1)
data_flat = -(data[3]-data[3][70000]) #+0.02*time)#coef4[0]*time
data_flat_smooth=smooth(data_flat,1000)
time_smooth=smooth(time,1000)
print(coef4[0])
plt.plot(time_smooth,1000*data_flat_smooth)
plt.grid(which='both')
plt.xlabel("temps (s)")
plt.ylabel("position (µm)")
plt.show()







