import numpy as np
import scipy
from tqdm import tqdm
import concurrent.futures
import time

import matplotlib.pyplot as plt

from Utils.Module_Plot.classPlot import *
from Utils.Module.acoustic_detection_librairy.Acoustic_detection_Library_opt import *
from Utils.Module.fct_synchro import *
from Utils.Module.acoustic_tools_corrected import *
from Utils.Module.fct_smooth import tsmooth

tdms_data = open_tdms_jules_acoustic("E:/2023-2024/2024-05-30_sm-acoustic/Sampling01_fs1MHz_ch3sm.tdms")
#plot = PaperPlot(remote=False)

acq_fr = 1e6
data = tdms_data[0]
t_data = np.arange(0, data.size, 1) / acq_fr
start = 69000000
end = 70000000  # 24000000
f_smooth = tsmooth(data, wt=0.00001, acq_fr=acq_fr)

##

n_decim=1

#fig, ax = plot.belleFigure('$t_b \ (s)$', '$fr \ (Hz)$', nfigure=None)
fig, ax = plt.subplots(1)
ax.plot(t_data[::n_decim], data[::n_decim], '.', ms=.5, alpha=.5, color='silver')
ax.plot(t_data[::n_decim], f_smooth[::n_decim], '-', linewidth=.5, ms=.5, alpha=1, color='C0')
# ax.plot(t_data[69269603], data[69269603], '*', color='tomato')
#plot.plt.grid(True, which='both')
#plot.plt.legend(loc="best")
ax.grid(which='both')

ax.set_xlabel("time (s)")
ax.set_ylabel("count")
ax.set_ylim([-300,300])


save = "E:/2023-2024/2024-05-30_sm-acoustic/raw_data"
#plot.fioritures(ax, fig, title=None, label=True, grid=None, save=save)

fig.savefig(save+".pdf")
fig.savefig(save+".svg")
fig.savefig(save+".png",dpi=600)
plt.close("all")

kkk=1

print("part {}\n".format(kkk))
kkk+=1


### params
fs = 10 ** 5
window_size = 2 ** 20
nseg = 512
noverlap = nseg - 1
Ncpu = 8

print("part {}\n".format(kkk))
kkk+=1

### def sub windows
start_sub_id = (np.arange(0, data.size, window_size)).astype(int)
end_sub_id = np.hstack((start_sub_id[1::], data.size))

sub_idx = [[i, j] for i, j in zip(start_sub_id, end_sub_id)]


# ------------------------------------------
def set_nrj_df(ini_idx, end_idx, data, methode='sum'):
    df = np.zeros(ini_idx.size, dtype=float)

    if ini_idx.size != 0:
        for bla in range(ini_idx.size):
            if methode == 'sum':
                try:
                    amplitude = np.sum(data[ini_idx[bla]:end_idx[bla]] ** 2)
                except TypeError:
                    print([ini_idx[bla], end_idx[bla]])
            else:
                amplitude = np.max(data[ini_idx[bla]:end_idx[bla]] ** 2)
            df[bla] = amplitude

    return df


def detect_evnts(i, sub_idx, data, threshold, acq_fr, nseg, noverlap, wt=0.00005):
    ini, end = sub_idx[i]
    logging.debug('{} {}'.format(ini, end))
    logging.debug('------- sous batch = {}'.format(ini))
    segment = data[ini:end]
    idx_shift = ini

    # Calculer la STFT pour le segment
    f, t, Zxx = my_stft(segment, fs, nperseg=nseg, noverlap=noverlap)
    # calcul STFT sur sub window
    # plt.pcolormesh(t, f, np.abs(Zxx))
    # plt.colorbar(label='Amplitude')
    # plt.show()
    stft = np.sum(np.abs(Zxx), axis=0)
    # trouver idx events (sur STFT size)
    index_beg, index_end, peak_idx = Find_peaks_easy(stft[:-1:], threshold)

    nrj_sum_signal = set_nrj_df(index_beg, index_end, segment, 'sum')
    nrj_sum_stft = set_nrj_df(index_beg, index_end, stft, 'sum')
    nrj_max_stft = set_nrj_df(index_beg, index_end, stft, 'max')
    dt = t[index_end] - t[index_beg]

    if index_beg.size != 0:
        logging.debug('found spike at idx {}'.format(index_beg))
        logging.debug('of didx = {}'.format(index_end - index_beg))
        logging.debug('of dt = {}'.format(t[index_end] - t[index_beg]))

        index_beg = index_beg + idx_shift
        index_end = index_end + idx_shift
        peak_idx = peak_idx + idx_shift

    return_dict = {'i': i,
                   'pos_idx': index_beg,
                   'neg_idx': index_end,
                   'peak_idx': peak_idx,
                   'nrj_sum_signal': nrj_sum_signal,
                   'nrj_sum_stft': nrj_sum_stft,
                   'nrj_max_stft': nrj_max_stft,
                   'dt': dt}

    return return_dict


from multiprocessing.pool import ThreadPool as Pool
from multiprocessing import freeze_support
from functools import partial

start_time = time.time()
dict = []
for i in tqdm(range(len(sub_idx))):
    dict.append(detect_evnts(i, sub_idx, data, 100, acq_fr, nseg, noverlap))
end_time = time.time()
print(f"pool: {end_time - start_time} seconds")

print("part {}\n".format(kkk))
kkk+=1


### les step sont dans l'ordre automatiquement !!!
result_dict = {'step': [],
               'pos_idx': [],
               'neg_idx': [],
               'peak_idx': [],
               'nrj_sum_signal': [],
               'nrj_sum_stft': [],
               'nrj_max_stft': [],
               'dt': []
               }
for l in tqdm(range(len(dict))):
    result_dict['step'].append(dict[l]['i'])
    result_dict['pos_idx'].append(dict[l]['pos_idx'])
    result_dict['neg_idx'].append(dict[l]['neg_idx'])
    result_dict['peak_idx'].append(dict[l]['peak_idx'])
    result_dict['nrj_sum_signal'].append(dict[l]['nrj_sum_signal'])
    result_dict['nrj_sum_stft'].append(dict[l]['nrj_sum_stft'])
    result_dict['nrj_max_stft'].append(dict[l]['nrj_max_stft'])
    result_dict['dt'].append(dict[l]['dt'])

# Optimisation avec np.fromiter
start_time = time.time()
steps = np.asarray(result_dict['step'], dtype=int)
pos_idx = np.fromiter((item for sublist in result_dict['pos_idx'] for item in sublist), dtype=int)
neg_idx = np.fromiter((item for sublist in result_dict['neg_idx'] for item in sublist), dtype=int)
peak_idx = np.fromiter((item for sublist in result_dict['peak_idx'] for item in sublist), dtype=int)
nrj_sum_signal = np.fromiter((item for sublist in result_dict['nrj_sum_signal'] for item in sublist), dtype=float)
nrj_sum_stft = np.fromiter((item for sublist in result_dict['nrj_sum_stft'] for item in sublist), dtype=float)
nrj_max_stft = np.fromiter((item for sublist in result_dict['nrj_max_stft'] for item in sublist), dtype=float)
dt = np.fromiter((item for sublist in result_dict['dt'] for item in sublist), dtype=float)
end_time = time.time()
print(f"np.fromiter: {end_time - start_time} seconds")

print("part {}\n".format(kkk))
kkk+=1


## Durées des events

yname = 'dt'
bla = dt[dt > 0]
fig, ax = plot.belleFigure('${} \ ({})$'.format(yname, ''),
                           '${}({})$'.format('Pdf_{log10}', yname), nfigure=None)
# calcul de la PDF
y_Pdf, x_Pdf = my_histo(bla, np.min(bla), np.max(bla),
                        'log', 'log', density=2, binwidth=None, nbbin=50)
ax.plot(x_Pdf, y_Pdf, '^', color='tomato')
plot.plt.xscale('log')
plot.plt.yscale('log')
save = "E:/2023-2024/2024-05-30_sm-acoustic/PDF_dt"
plot.fioritures(ax, fig, title='Durées des évenements', label=True, grid=None, save=save, major=None)

print("part {}\n".format(kkk))
kkk+=1


## MAG 1 (sum in time)

yname = 'NRJ'
bla = nrj_sum_signal[nrj_sum_signal > 0]
fig, ax = plot.belleFigure('${} \ ({})$'.format(yname, ''),
                           '${}({})$'.format('Pdf_{log10}', yname), nfigure=None)
# calcul de la PDF
y_Pdf, x_Pdf = my_histo(bla, np.min(bla), np.max(bla),
                        'log', 'log', density=2, binwidth=None, nbbin=50)
# result, xs, n_f, perr = fit_powerlaw(x_Pdf, y_Pdf, n_cutoff=1e-20, start_x=3e4, plot=True, f_scale=1.0,
#                                      function_to_fit='power')
# ax.plot(xs, power_fit(xs, *result.x), 'k--', label='$a={} \pm {}$'.format(np.round(result.x[1], 2),
#                                                                           np.round(perr[1], 2)))
ax.plot(x_Pdf, y_Pdf, '^', color='tomato')
plot.plt.xscale('log')
plot.plt.yscale('log')
save = "E:/2023-2024/2024-05-30_sm-acoustic/PDF_nrj_sumTime"
plot.fioritures(ax, fig, title='Distribution des magnitudes', label=True, grid=None, save=save, major=None)

print("part {}\n".format(kkk))
kkk+=1


## MAG 2 (sum in freq)

yname = 'NRJ'
bla = nrj_sum_stft[nrj_sum_stft > 0]
fig, ax = plot.belleFigure('${} \ ({})$'.format(yname, ''),
                           '${}({})$'.format('Pdf_{log10}', yname), nfigure=None)
# calcul de la PDF
y_Pdf, x_Pdf = my_histo(bla, np.min(bla), np.max(bla),
                        'log', 'log', density=2, binwidth=None, nbbin=50)
# result, xs, n_f, perr = fit_powerlaw(x_Pdf, y_Pdf, n_cutoff=1e-20, start_x=3e6, plot=True, f_scale=1.0,
#                                      function_to_fit='power')
# ax.plot(xs, power_fit(xs, *result.x), 'k--', label='$a={} \pm {}$'.format(np.round(result.x[1], 2),
#                                                                           np.round(perr[1], 2)))
ax.plot(x_Pdf, y_Pdf, '^', color='tomato')
plot.plt.xscale('log')
plot.plt.yscale('log')
save = "E:/2023-2024/2024-05-30_sm-acoustic/PDF_nrj_sumFreq"
plot.fioritures(ax, fig, title='Distribution des magnitudes', label=True, grid=None, save=save, major=None)

print("part {}\n".format(kkk))
kkk+=1


## MAG 3 (max in freq)

yname = 'NRJ'
bla = nrj_max_stft
fig, ax = plot.belleFigure('${} \ ({})$'.format(yname, ''),
                           '${}({})$'.format('Pdf_{log10}', yname), nfigure=None)
# calcul de la PDF
y_Pdf, x_Pdf = my_histo(bla, np.min(bla), np.max(bla),
                        'log', 'log', density=2, binwidth=None, nbbin=40)
result, xs, n_f, perr = fit_powerlaw(x_Pdf, y_Pdf, n_cutoff=1e-20, start_x=9e3, plot=True, f_scale=1.0,
                                     function_to_fit='power')
ax.plot(xs, power_fit(xs, *result.x), 'k--', label='$a={} \pm {}$'.format(np.round(result.x[1], 2),
                                                                          np.round(perr[1], 2)))
ax.plot(x_Pdf, y_Pdf, '^', color='tomato')
plot.plt.xscale('log')
plot.plt.yscale('log')
save = "E:/2023-2024/2024-05-30_sm-acoustic/PDF_nrj_sumFreq"
plot.fioritures(ax, fig, title='Distribution des magnitudes', label=True, grid=None, save=save, major=None)

print("part {}\n".format(kkk))
kkk+=1


## Temps inter event
# ------------------------------------------
def set_dt_btw_nzero(t, norm=True):
    logging.info('------- Set dt_btw')
    dt_btw = np.round(t[1::] - t[0:-1:], int(np.log10(acq_fr)))
    if norm:
        dt_btw = dt_btw / np.mean(dt_btw)
    return dt_btw


seuil = 12000# 160000
dt_big = dt[nrj_max_stft > seuil]
df = nrj_max_stft.copy()[nrj_max_stft > seuil]
dt_btw = set_dt_btw_nzero(peak_idx[nrj_max_stft > seuil] / acq_fr, norm=True)

f = dt_btw[dt_btw > 0]
min_f = np.min(f)
start_x = np.min(f)
stop = 1e-10  ## 5e-6
nbbin = 40
yname = '\Delta t_{btw}'
fig, ax = plot.belleFigure('${}/ <{} >\ (s)$'.format(yname, yname),
                           '${}({})$'.format('Pdf_{log10}', yname), nfigure=None)
ax.plot(dt_btw, 0.5 * dt_btw ** (-0.33) * np.exp(-dt_btw / 1.59), '.', color='k')
# ax.plot(dt_btw, 0.5*dt_btw**(0)*np.exp(-dt_btw/1.59), '.', color='r')
y_Pdf, x_Pdf = my_histo(f, np.min(f), np.max(f),
                        'log', 'log', density=2, binwidth=None,
                        nbbin=nbbin)
ax.plot(x_Pdf, y_Pdf, '^', color='tomato')
ax.axvline(x=4, c='silver', linestyle='--')
# plot.plt.ylim(1e-7, 10)
plot.plt.xscale('log')
plot.plt.yscale('log')
save = "E:/2023-2024/2024-05-30_sm-acoustic/dt_btw_max_stft"
plot.fioritures(ax, fig, title='Distribution des temps inter-evenements', label=True, grid=None, save=save, major=None)

print("part {}\n".format(kkk))
kkk+=1

