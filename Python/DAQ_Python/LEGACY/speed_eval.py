import numpy as np
import matplotlib.pyplot as plt

x=[2.5e-2,5e-2,7.5e-2,1e-1]

t=np.array([[-6.7e-05, -4.7e-05, -1.8e-05,  1.8e-05],
            [-8.0e-05, -3.2e-05, -9.0e-06,  1.8e-05],
            [-6.6e-05, -4.4e-05, -1.2e-05,  2.3e-05],
            [-7.2e-05, -3.7e-05, -1.8e-05,  1.7e-05],
            [-6.8e-05, -2.8e-05, -4.0e-06,  2.4e-05]])

fig, axes = plt.subplots(1, 5,sharex=True,sharey=True)

for i in range(5):
    axes[i].plot(x,t[i])
    a,b=np.polyfit(x,t[i],deg=1)
    axes[i].plot(x,a*np.array(x)+b,label="v={}".format(int(1/a)))
    print(1/a)
    axes[i].legend()
    axes[i].grid(which='both')


plt.show()


