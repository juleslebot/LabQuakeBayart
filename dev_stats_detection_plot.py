import numpy as np
import matplotlib.pyplot as plt
from Utils.Module_Plot.classPlot import *

acq_fr = 1e6

loc = 'E:/2023-2024/2024-05-30_sm-acoustic/'

pos_idx = np.load(loc+'save_threshold100/pos_idx.npy',allow_pickle=True)
neg_idx = np.load(loc+'save_threshold100/neg_idx.npy',allow_pickle=True)
peak_idx = np.load(loc+'save_threshold100/peak_idx.npy',allow_pickle=True)
nrj_sum_signal = np.load(loc+'save_threshold100/nrj_sum_signal.npy',allow_pickle=True)
nrj_sum_stft = np.load(loc+'save_threshold100/nrj_sum_stft.npy',allow_pickle=True)
nrj_max_stft = np.load(loc+'save_threshold100/nrj_max_stft.npy',allow_pickle=True)
dt = np.load(loc+'save_threshold100/dt.npy',allow_pickle=True)

## Durées des events

yname = 'dt'
bla = dt[dt > 0]
fig, ax = plt.subplots(1)
# calcul de la PDF
y_Pdf, x_Pdf = my_histo(bla, np.min(bla), np.max(bla),
                        'log', 'log', density=2, binwidth=None, nbbin=50)
ax.plot(x_Pdf, y_Pdf, '^', color='tomato')
ax.set_xscale('log')
ax.set_yscale('log')
save = loc+'PDF_dt'


ax.grid(which='both')
ax.set_ylabel(yname)

fig.suptitle("Durées des évenements")
fig.savefig(save+".pdf")
fig.savefig(save+".png",dpi=1200)
plt.show()
plt.close("all")


## MAG 1 (sum in time)

yname = 'NRJ'
bla = nrj_sum_signal[nrj_sum_signal > 0]
fig, ax = plt.subplots(1)

# calcul de la PDF
y_Pdf, x_Pdf = my_histo(bla, np.min(bla), np.max(bla),
                        'log', 'log', density=2, binwidth=None, nbbin=50)
result, xs, n_f, perr = fit_powerlaw(x_Pdf, y_Pdf, n_cutoff=1e-20, start_x=3e4, plot=True, f_scale=1.0,
                                     function_to_fit='power')
ax.plot(xs, power_fit(xs, *result.x), 'C0--', label='$B={} \pm {}$'.format(np.round(result.x[1], 2),
                                                                          np.round(perr[1], 2)))

ax.plot(x_Pdf, y_Pdf, '^', color='C0')
ax.set_xscale('log')
ax.set_yscale('log')
save = loc+'PDF_nrj_sumTime'
ax.grid(which='both')
ax.set_ylabel(yname)
ax.legend()

fig.suptitle("Distribution des magnitudes")
fig.savefig(save+".pdf")
fig.savefig(save+".png",dpi=1200)
plt.show()
plt.close("all")



## MAG 2 (sum in freq)

yname = 'NRJ'
bla = nrj_sum_stft[nrj_sum_stft > 0]
fig, ax = plt.subplots(1)
# calcul de la PDF
y_Pdf, x_Pdf = my_histo(bla*.05, np.min(bla), np.max(bla),
                        'log', 'log', density=2, binwidth=None, nbbin=50)

result, xs, n_f, perr = fit_powerlaw(x_Pdf, y_Pdf, n_cutoff=1e-20, start_x=3e6, plot=True, f_scale=1.0,
                                     function_to_fit='power')
ax.plot(xs, power_fit(xs, *result.x), 'C1--', label='$B={} \pm {}$'.format(np.round(result.x[1], 2),
                                                                          np.round(perr[1], 2)))
ax.plot(x_Pdf, y_Pdf, '^', color='C1')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()

save = loc+'PDF_nrj_sumFreq'

ax.grid(which='both')
ax.set_ylabel(yname)

fig.suptitle("Distribution des magnitudes")
fig.savefig(save+".pdf")
fig.savefig(save+".png",dpi=1200)
plt.show()
plt.close("all")



## MAG 3 (max in freq)

yname = 'NRJ'
bla = nrj_max_stft
fig, ax = plt.subplots(1)
# calcul de la PDF
y_Pdf, x_Pdf = my_histo(bla, np.min(bla), np.max(bla),
                        'log', 'log', density=2, binwidth=None, nbbin=40)
result, xs, n_f, perr = fit_powerlaw(x_Pdf, y_Pdf, n_cutoff=1e-20, start_x=9e3, plot=True, f_scale=1.0,
                                     function_to_fit='power')
ax.plot(xs, power_fit(xs, *result.x), 'k--', label='$B={} \pm {}$'.format(np.round(result.x[1], 2),
                                                                          np.round(perr[1], 2)))
ax.plot(x_Pdf, y_Pdf, '^', color='C2')
ax.legend()
save = loc+'PDF_nrj_maxFreq'



ax.set_xscale('log')
ax.set_yscale('log')

ax.grid(which='both')
ax.set_ylabel(yname)

fig.suptitle("Distribution des magnitudes")
fig.savefig(save+".pdf")
fig.savefig(save+".png",dpi=1200)
plt.show()
plt.close("all")




## Temps inter event
# ------------------------------------------
def set_dt_btw_nzero(t, norm=True):
    logging.info('------- Set dt_btw')
    dt_btw = np.round(t[1::] - t[0:-1:], int(np.log10(acq_fr)))
    if norm:
        dt_btw = dt_btw / np.mean(dt_btw)
    return dt_btw


seuil = 16000
yname = '\Delta t_{btw}'
fig, ax = plt.subplots(1)


dt_big = dt[nrj_max_stft > seuil]
df = nrj_max_stft.copy()[nrj_max_stft > seuil]
dt_btw = set_dt_btw_nzero(peak_idx[nrj_max_stft > seuil] / acq_fr, norm=True)

f = dt_btw[dt_btw > 0]
min_f = np.min(f)
start_x = np.min(f)
stop = 1e-10  ## 5e-6
nbbin = 40
x = np.arange(np.min(dt_btw),np.max(dt_btw),.01)
theta =  x ** (-0.33) * np.exp(-x / 1.5)
ax.plot(x,theta, '-', color='k')
# ax.plot(dt_btw, 0.5*dt_btw**(0)*np.exp(-dt_btw/1.59), '.', color='r')
y_Pdf, x_Pdf = my_histo(f, np.min(f), np.max(f),
                        'log', 'log', density=2, binwidth=None,
                        nbbin=nbbin)
ax.plot(x_Pdf, y_Pdf, '.',label=f'E0 = {seuil}'.format(),alpha=.5)

ax.axvline(x=4, c='silver', linestyle='--')
# plot.plt.ylim(1e-7, 10)
ax.set_xscale('log')
ax.set_yscale('log')
save = loc+'dt_btw_max_stft'

ax.grid(which='both')
ax.set_ylabel(yname)
ax.legend()

fig.suptitle("Distribution des temps inter-evenements")
fig.savefig(save+".pdf")
fig.savefig(save+".png",dpi=1200)
plt.show()
plt.close("all")




