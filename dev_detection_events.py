import numpy as np
import scipy
from tqdm import tqdm
import matplotlib.pyplot as plt

from Python.Utils.Module_Plot.classPlot import *
from Python.Utils.Module.acoustic_detection_librairy.Acoustic_detection_Library_opt import *
from Python.Utils.Module.fct_synchro import *
from Python.Utils.Module.acoustic_tools_corrected import *
from Python.Utils.Module.fct_smooth import tsmooth

loc = 'C:/Users/Manips/Desktop/tmp/2024-06-25_full_manip/01/'
tdms_data = open_tdms_jules_acoustic(loc + "Sampling01.tdms")
plot = PaperPlot(remote=False)

fr_acq = 1e6
start = 23000000
end = 24000000
data = tdms_data[0][start:end]
t_data = np.arange(0, data.size, 1)/fr_acq

f_smooth = tsmooth(data, wt=0.000005, acq_fr=fr_acq)
fig, ax = plot.belleFigure('$t_b \ (s)$', '$fr \ (Hz)$', nfigure=None)
ax.plot(t_data, data, '.', color='silver')
ax.plot(t_data, f_smooth, '.', color='C2')
plot.plt.grid(True, which='both')
save = None
plot.fioritures(ax, fig, title=None, label=True, grid=None, save=save)

### params
acq_fr = 1e6
window_size = 2 ** 20  # au plus grand possible vu la RAM
nseg = 1024 # nbre pts / fft
noverlap = nseg - 1
nbin = 50000

### def sub windows
start_sub_id = (np.arange(0, data.size, window_size)).astype(int)
end_sub_id = np.hstack((start_sub_id[1::], data.size))

sub_idx = [[i, j] for i, j in zip(start_sub_id, end_sub_id)]

### calcul Zxx sur sub window
idx = None
ini = 0 #idx[0]
end = data.size #idx[1]
logging.debug('{} {}'.format(ini, end))
segment = data[ini:end]
segment_smooth = tsmooth(segment, wt=0.000005, acq_fr=fr_acq)
t_segment = t_data[ini:end] - t_data[ini]
idx_shift = ini
t_shift = t_data[ini]
logging.debug('------- sous batch start at = {} idx'.format(idx_shift))
logging.debug('------- sous batch start at = {} t'.format(t_shift))

# Calculer la STFT pour le segment
f, t, Zxx_s = my_stft(segment_smooth, acq_fr, nperseg=nseg, noverlap=noverlap)

yname = 'Zxx'
bla = np.abs(Zxx_s).reshape(Zxx_s.size)
fig, ax = plot.belleFigure('${} \ ({})$'.format(yname, ''),
                           '${}({})$'.format('Pdf_{log10}', yname), nfigure=None)
### calcul de la PDF
y_Pdf, x_Pdf = my_histo(bla, np.min(bla), np.max(bla),
                        'log', 'log', density=2, binwidth=None, nbbin=50)
ax.plot(x_Pdf, y_Pdf, '^', color='tomato')
plot.plt.xscale('log')
plot.plt.yscale('log')
save = None
plot.fioritures(ax, fig, title='Histogramme', label=True, grid=None, save=save, major=None)

t_sample = np.arange(0, t.size, 100)
# Pour un plot 2D utiliser noverlap = 750 sinon ça plante
fig, ax = plot.belleFigure('$t \ (s)$', '$Fr \ (Hz)$', figsize=(10, 6))
plot.plt.pcolormesh(t[t_sample], f, np.abs(Zxx_s[0:20000, t_sample]), vmin=0, vmax=10)
plt.colorbar(label='Amplitude')
save = None
plot.fioritures(ax, fig, title='Spectrogramme', label=True, grid=None, save=save)

### calcul STFT sur sub window
stft_s = np.sum(np.abs(Zxx_s), axis=0)

fig, ax = plot.belleFigure('$t_b \ (s)$', '$fr \ (Hz)$', nfigure=None)
ax.plot(t_segment, segment, '.', color='silver')
ax.plot(t, stft_s, '.', color='C0')
plot.plt.grid(True, which='both')
save = None
plot.fioritures(ax, fig, title='Amplitude et Energie', label=True, grid=None, save=save)

yname = 'stft_s'
bla = stft_s
fig, ax = plot.belleFigure('${} \ ({})$'.format(yname, ''),
                           '${}({})$'.format('Pdf_{log10}', yname), nfigure=None)
### calcul de la PDF
y_Pdf, x_Pdf = my_histo(bla, np.min(bla), np.max(bla),
                        'log', 'log', density=2, binwidth=None, nbbin=50)
ax.plot(x_Pdf, y_Pdf, '^', color='tomato')
plot.plt.xscale('log')
plot.plt.yscale('log')
save = None
plot.fioritures(ax, fig, title=None, label=True, grid=None, save=save, major=None)

pos_idx, neg_idx, peak_idx = Find_peaks_easy(stft_s[:-1:], threshold=70)

if pos_idx.size != 0:
    logging.warning('found spike at idx {}'.format(pos_idx))
    logging.warning('of didx = {}'.format(neg_idx - pos_idx))
    logging.warning('of dt = {}'.format(t[neg_idx] - t[pos_idx]))

fig, ax = plot.belleFigure('$t_b \ (s)$', '$fr \ (Hz)$', nfigure=None)
ax.plot(t_segment + t_shift, segment, '.', color='C0')
ax.plot(t_segment + t_shift, segment_smooth, '.', color='C2')
for e in range(pos_idx.size):
    ax.plot(t_data[pos_idx[e]:neg_idx[e]], data[pos_idx[e]:neg_idx[e]], '.', color='C1')
    ax.plot(t_data[peak_idx[e]], stft_s[peak_idx[e]], '*', color='C0')
plot.plt.grid(True, which='both')
plot.plt.legend(loc="best")
save = None
plot.fioritures(ax, fig, title=None, label=True, grid=None, save=save)


logging.warning('for overlap = {} we find {} events'.format(noverlap, pos_idx.size))
