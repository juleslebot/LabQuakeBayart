import matplotlib.pyplot as plt
import numpy as np
import logging
import Python.DAQ_Python.Python_DAQ as daq
from sklearn.linear_model import LinearRegression
from mpl_point_clicker import clicker
from scipy.signal import butter, sosfilt, sosfreqz
#from scipy import signal

def signal_to_tensor(data):
# input : data[gages]
# output : (eps_xx,eps_xy,eps_yy)
    for i in range(data.shape[0]//3):
        ch_1=data[3*i]
        ch_2=data[3*i+1]
        ch_3=data[3*i+2]
        ch_1,ch_2,ch_3=daq.voltage_to_strains(ch_1,ch_2,ch_3,amp=2000)
        ch_1,ch_2,ch_3=daq.rosette_to_tensor(ch_1,ch_2,ch_3)
        data[3*i]=ch_1
        data[3*i+1]=ch_2
        data[3*i+2]=ch_3

    eps_yy = data[1::3]
    eps_xy = data[2::3]
    eps_xx = data[0::3]
    return eps_xx,eps_xy,eps_yy

# def onpick(event):
#     x_coord = event.mouseevent.xdata
#     y_coord = event.mouseevent.ydata
#     ax=event.mouseevent.inaxes
#     row=(ax.get_subplotspec().rowspan.start)
#     col=(ax.get_subplotspec().colspan.start)
#     print(f'Picked point: ({x_coord:.2f}, {y_coord:.2f})')
#     indexes[row,col]=x_coord

def propagation_clicker(data,freq_s, layout=(4,5),a=300):
    # pass band (ou bas) avant de donner les données ici
    time = np.arange(0,data.shape[-1]-a+1,1)/freq_s
    figX,figY = layout[0],layout[1]
    indexes = np.zeros((figX,figY,2,2))
    if figX*figY != data.shape[0]:
        logging.error('Les parametres de plots sont pour {} traces de sigma'.format(figX*figY))
        pass

    # low = .0001
    # high = 5000
    # filter = butter(5, [2*low/freq_s, 2*high/freq_s], analog=False, btype='band', output='sos')
    # data = sosfilt(filter, data)

    data = daq.smooth(data,a)

    fig, axs = plt.subplots(figX,figY,sharex=True,sharey=False)
    klicker = [[ [] for i in range(layout[1])] for i in range(layout[0])]
    for i in range(len(data)):
        #f, t, Sxx = signal.spectrogram(data[i,:], freq_s)
        #axs[i//figY][i%figY].pcolormesh(t, f, Sxx)
        axs[i//figY][i%figY].plot(time,data[i,:], picker=True, pickradius=6)
        axs[i//figY][i%figY].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        axs[i//figY][i%figY].grid("both")
        axs[i//figY][i%figY].set_title('Jauge {}'.format(i+1))
        klicker[i//figY][i%figY] = clicker(axs[i//figY][i%figY],['before','after'],markers=["x",'+'])
    axs[-1][0].set_xlabel('time (s)')
    fig.set_size_inches(14,8)
    plt.show()

    for j in range(figY):
        for i in range(figX):
            try :
                indexes[i,j,0,:] = klicker[i][j].get_positions()['before'][0]
                indexes[i,j,1,:] = klicker[i][j].get_positions()['after'][0]

            except IndexError: pass
    delta_sigma = (indexes[:,:,1,1]-indexes[:,:,0,1]).reshape(data.shape[0],)
    t_arr = indexes[:,:,0,0].reshape(data.shape[0],)

    return delta_sigma,t_arr

def velocity_clicker(rosette_pos,indexes):
    fig, ax = plt.subplots()
    ax.plot(rosette_pos,indexes,'k.')
    ax.set_xlabel('distance along the interface (m)')
    ax.set_ylabel('time (s)')
    fig.suptitle('Pick at least two points to compute a velocity')
    klicker = clicker(ax,['event'], markers=["x"])
    fig.set_size_inches(14,8)
    plt.show()
    points = klicker.get_positions()['event']
    try :
        reg = np.polyfit(points[:,0],points[:,1],1)
        c = reg[0]
    except IndexError:
        c = 0
        logging.warning('Pas de vitesse calculee, valeur de base : 0')
    return c

def detect_stress_drop(eps_xx,eps_xy,eps_yy,rosette_pos,freq_s=1e+6):
    sigma_xx,sigma_yy,sigma_xy = daq.eps_to_sigma(eps_xx,eps_yy,eps_xy)
    n_ros = eps_xx.shape[0]
    #delta_sigma_xy_click,t_arr = propagation_clicker(sigma_xy,freq_s)
    #c_rupture = velocity_clicker(rosette_pos,t_arr)
    delta_sigma_xy_moy = np.zeros(n_ros)
    for i in range(n_ros):
        # drop_idx = int(indexes[i]*freq_s)
        sigma_xy_before = np.mean(sigma_xy[i,:5000])
        sigma_xy_after = np.mean(sigma_xy[i,-5000:])
        delta_sigma_xy_moy[i] = sigma_xy_before - sigma_xy_after
    dx = rosette_pos[1]-rosette_pos[0]
    sigma_yy_dx = np.sum(sigma_yy*dx,axis=0)
    sigma_xy_dx = np.sum(sigma_xy*dx,axis=0)
    #return delta_sigma_xy_moy,delta_sigma_xy_click, c_rupture,sigma_yy_dx,sigma_xy_dx
    return delta_sigma_xy_moy,sigma_yy_dx,sigma_xy_dx

def click_stress_drop(eps_xx,eps_yy,eps_xy,rosette_pos,freq_s=1e+6):
    sigma_xx,sigma_yy,sigma_xy = daq.eps_to_sigma(eps_xx,eps_yy,eps_xy)
    delta_sigma_xy_click,t_arr = propagation_clicker(sigma_xy,freq_s)
    c_rupture = velocity_clicker(rosette_pos,t_arr)
    return delta_sigma_xy_click,c_rupture




def nrj(event, method='sum'):
    if method == 'sum':
        nrj = np.sum(event**2,axis=0) # verifier qu'on somme bien en temps
    elif method == 'max':
        nrj = np.max(event**2,axis=0) # idem
    else: pass # Warning
    return nrj

def convert(data):
    # a importer proprement...
    gages_ch = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62])
    #=========================================
    for i in range(len(locals()['gages_ch'])//3):
        ch_1=data[3*i]
        ch_2=data[3*i+1]
        ch_3=data[3*i+2]
        ch_1,ch_2,ch_3=daq.voltage_to_strains(ch_1,ch_2,ch_3,amp=2000)
        ch_1,ch_2,ch_3=daq.rosette_to_tensor(ch_1,ch_2,ch_3)
        data[3*i]=ch_1
        data[3*i+1]=ch_2
        data[3*i+2]=ch_3
    return data

def treat_event(file):
    # a importer proprement...
    gages_on_both_side = False
    gages_ch = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62])
    rosette_pos = np.linspace(0,0.1425,20)
    #=========================================


    file_zero = file[:-13]+"event-001.npy"
    try : data_zero = np.load(file_zero,allow_pickle=True)
    except FileNotFoundError:
        logging.warning('No such file: {}\n'.format(file_zero))
        file_zero = input('Correct path to file_zero: ')
        data_zero = np.load(file_zero,allow_pickle=True)
    try : data_acq = np.load(file,allow_pickle=True)
    except FileNotFoundError:
        logging.warning('No such file: {}\n'.format(file))
        file = input('Correct path to file: ')
        data_acq = np.load(file,allow_pickle=True)
    data_acq = daq.smooth(data_acq,10) # smoothing data
    data_acq = np.transpose(np.transpose(data_acq)-np.mean(data_zero,axis=1)) # correcting systematic shift

    fs = daq.voltage_to_force(data_acq[31,:])
    fn = daq.voltage_to_force(data_acq[15,:])
    acoustic = data_acq[63,:]

    data_gages = data_acq[gages_ch]
    eps_xx,eps_xy,eps_yy = signal_to_tensor(data_gages)
    if gages_on_both_side:
        print('Gages on both side : sign correction')
        eps_xy[1::2,:] = -eps_xy[1::2,:]
    mu = fs.mean() / fn.mean()
    # delta_sigma_xy_moy, delta_sigma_xy_click, c_rupture, sigma_yy_dx, sigma_xy_dx = detect_stress_drop(eps_xx,eps_xy,eps_yy,rosette_pos)
    delta_sigma_xy_moy,sigma_yy_dx,sigma_xy_dx = detect_stress_drop(eps_xx,eps_xy,eps_yy,rosette_pos)
    nrj_sum_signal = nrj(data_acq[gages_ch],'sum')
    nrj_max_stft = nrj(data_acq[gages_ch],'max')

    event_dict = {
            'fn' : fn,
            'fs' : fs,
            'trig' : data_acq[47,:],
            'acoustic' : acoustic,
            'mu' : mu,
            #'rupture_velocity': c_rupture,
            'fn_gages':sigma_yy_dx,
            'fs_gages': sigma_xy_dx}
    ch = []
    for i in range(len(rosette_pos)):
        ch.append(  [eps_xx[i,:],
                    eps_xy[i,:],
                    eps_yy[i,:],
                    delta_sigma_xy_moy[i],
                    #delta_sigma_xy_click[i],
                    nrj_sum_signal[i],
                    nrj_max_stft[i]])

    return event_dict,ch
