'''
Plusieurs petiots traitement sur les donnees complet.
Recalage en temps du SM et de l'acquisition acoustique.
Abandonne au profit de dev_stat_event_* pour le traitement des donnees acoustiques.
'''
path = "E:/2023-2024/2024-05-30_sm-acoustic/"
filetdms = 'Sampling01_fs1MHz_ch3sm.tdms'
filepath = path+filetdms
fech_ch = 1e+6
ch1,ch2,ch3 = open_data(filepath,'cont')
A = 1.0245
B =  70.7521
time_ch = np.arange(0,len(ch1)/fech_ch,1/fech_ch) + 74.798
ch = ch2

sm = np.load(path+"slowmon.npy",allow_pickle=True)
time_sm = np.load(path+"slowmon_time.npy")

fn_sm = sm[15,:]
fs_sm = sm[31,:]
ac_sm = sm[63,:]
trigger = 100*sm[47,:]

fech_sm = len(time_sm)/(time_sm[-1]-time_sm[0])

def convert_index_sm_to_ch(index,Dt=231.66-156.97,fech_ch=fech_ch,fech_sm=fech_sm):
    return int(((index/fech_sm)-Dt)*fech_ch)

def convert_index_ch_to_sm(index,Dt=231.66-156.97,fech_ch=fech_ch,fech_sm=fech_sm):
    return int(((index/fech_ch)+Dt)*fech_sm)

##

energy = magnitude_distribution(ch,threshold=50)
n,bins,patches = plt.hist(energy)
plt.show()
plt.loglog(bins[:-1],n,'+')
plt.show()
##
Nevent_s = np.zeros(100)
for i,s in enumerate(np.arange(0,100,1)):
    Nevent_s[i] = len(look_for_event(ch,10000,s))

plt.plot(np.arange(0,100,1),Nevent_s)
plt.xlabel('Seuil (count)')
plt.ylabel('Nevent')
plt.title("Nombre d'evenement detecté en fonction du seuil de detection" )
plt.show()

## Filter ch low pass

fc = 100000
Wn= 2*fc/fech_ch
N = 10 # order of the filter (à quelle point la coupure est sharp je crois)
b, a = scipy.signal.butter(N, Wn, 'low')
ch1_filt = scipy.signal.filtfilt(b, a, ch1)

##
time_ac = np.linspace(0,len(ch1)/1e+6,len(ch1))
f_ech_sm = len(time_sm)/(time_sm[-1]-time_sm[0])
f_ech_ac = 1e+6
#A = 1/0.9931
#B = 98.055
A = 1
B = 59.509

plt.plot(A*(time_sm-B),100*fs_sm,label='Fs')
plt.plot(A*(time_sm-B),trigger,label='Trig')
plt.plot(A*(time_sm-B),ac_sm,label='Acoustique sm')
plt.plot(time_ac-23.095,ch1/5,label='Acoustique')
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Force [N], Count')
plt.show()
plt.close('all')

##
index = look_for_event(ch,10000,50)
for ind in index:
    data = ch[ind[0]:ind[1]]
    f,t,Sxx = scipy.signal.spectrogram(data,1e+6,nperseg=100,noverlap=25)
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    #plt.show()

##
Nevent = len(index)
event = np.zeros((Nevent,1700))
energy = np.zeros(Nevent)
DFs = np.zeros(Nevent)
index_event_ch = np.zeros(Nevent)
for i,ind in enumerate(index):
    data = ch[ind[0]:ind[1]]
    Nstart, Nend = index_event(data,200,1500)
    if Nstart <0:
        event[i,:] = np.concatenate((ch[ind[0]+Nstart:ind[0]],data[0:Nend]))
    elif Nend > 10000:
        event[i,:] = np.concatenate((data[Nstart:],ch[ind[1]:ind[0]+Nend]))
    else :
        event[i,:] = data[Nstart:Nend]
    ind_sm_0 = convert_index_ch_to_sm(ind[0])
    ind_sm_1 = convert_index_ch_to_sm(ind[1])
    Fs_event = fs_sm[ind_sm_0:ind_sm_1]
    DFs[i] = (np.max(Fs_event)-np.min(Fs_event))
    energy[i] = np.sum(event[i,:]**2)
    index_event_ch[i] = ind[0]+Nstart+200

plt.plot(DFs,energy,'+')
plt.xlabel('\Delta F_s')
plt.ylabel('Energie acoustique')
plt.show()

##

# delta t entre deux evenements de magnitude sup à un seuil
Mw_lim = 25e+6
index_big_event = []

for i,e in enumerate(energy):
    if e >= Mw_lim:
        index_big_event.append(index_event_ch[i])

dt_aftershock = [[] for n in range(len(index_big_event))]

for i in range(len(index_big_event)-1):
    print(f'\ni={i}'.format())
    index_aftershocks = np.intersect1d(np.where(index_event_ch > index_big_event[i]), np.where(index_event_ch < .5*index_big_event[i] + .5*index_big_event[i+1]))
    for ii in index_aftershocks:
        dt_aftershock[i].append(index_event_ch[ii]-index_big_event[i])

# cas i = len(index_big_event) à traiter independamment

## Camera
event_cam = [5517:5520], [6782:6786], [7714:7716], [9514:9516], [9874,9875], [10008:10011], [10563:10592],
    [11129:11131], [11986:11988], [12098:12156], [12619:12622]