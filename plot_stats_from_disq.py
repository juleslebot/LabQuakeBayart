'''
Petit programme pour ploter ensemble les distributions de magnitudes QUI SONT DEJA CALCULEES et sauvegardees sur le disque.
'''
import numpy as np
import matplotlib.pyplot as plt
from Python.Utils.Module_Plot.classPlot import *

loc = 'C:/Users/Manips/Desktop/tmp/2024-06-25_full_manip/'
data = [np.load(loc + '01/save_averageOFF_thr70/nrj_max_stft.npy',allow_pickle=True),
        np.load(loc + '02/save_averageOFF_thr70/nrj_max_stft.npy',allow_pickle=True),
        np.load(loc + '03/save_avOFF_70/nrj_max_stft.npy',allow_pickle=True),
        np.load(loc + '04_nocamera/save_averageOFF_70/nrj_max_stft.npy',allow_pickle=True)]

yname = 'NRJ'
fig, ax = plt.subplots(1)
legends = ('Fn=300Kg', 'Fn=400Kg', 'Fn=200Kg','Fn=250Kg')
colors = ('C0','C1','C2','C3')
# calcul de la PDF
for i,bla in enumerate(data):
    y_Pdf, x_Pdf = my_histo(bla, np.min(bla), np.max(bla),'log', 'log', density=2, binwidth=None, nbbin=40)
    result, xs, n_f, perr = fit_powerlaw(x_Pdf, y_Pdf, n_cutoff=1e-20, start_x=9e3, plot=True, f_scale=1.0,function_to_fit='power')
    ax.plot(xs, power_fit(xs, *result.x), '--',label='$a={} \pm {}$'.format(np.round(result.x[1], 2), np.round(perr[1], 2)), c=colors[i])
    ax.plot(x_Pdf, y_Pdf, '.', color=colors[i], label=legends[i])
    save = loc+"PDF_nrj_maxFreq"

y_Pdf, x_Pdf = my_histo(np.concatenate((data[0],data[1],data[2],data[3])), np.min(bla), np.max(bla),'log', 'log', density=2, binwidth=None, nbbin=40)
result, xs, n_f, perr = fit_powerlaw(x_Pdf, y_Pdf, n_cutoff=1e-20, start_x=9e3, plot=True, f_scale=1.0,function_to_fit='power')
ax.plot(xs, power_fit(xs, *result.x), 'k-.',label='$a={} \pm {}$'.format(np.round(result.x[1], 2), np.round(perr[1], 2)))
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()

ax.grid(which='both')
ax.set_ylabel(yname)

fig.suptitle("Distribution des magnitudes")
plt.show()
plt.close("all")
