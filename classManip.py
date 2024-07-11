import logging
import numpy as np
import glob
from tqdm import tqdm
import os.path
import sys

import Python.DAQ_Python.Python_DAQ as daq
import Python.DAQ_Python.bib_fastacq_jules as facq
 ##

gages_ch = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62])

class channel():
    def __init__(self,*args):
        arg = args[0]
        if type(arg) == list:
            self.create_from_file(arg)
        elif type(arg) == dict:
            self.convert_dict_to_ch(arg)
        else:
            logging.error('Error: Argument type {} is not supported for channel.__init__()'.format(type(arg)))
            print(len(arg))


    def create_from_file(self,data):
        eps_xx,eps_xy,eps_yy,delta_sigma_xy_moy,nrj_sum,nrj_max = data

        self.strain = np.array((eps_xx,eps_yy,eps_xy))
        self.delta_sigma_xy_moy = delta_sigma_xy_moy
        self.nrj_sum = nrj_sum
        self.nrj_max = nrj_max
        # self.position = position
        pass

    def convert_ch_to_dict(self):
        ch_dict = {
            'eps_xx' : self.strain[0,:],
            'eps_xy' : self.strain[1,:],
            'eps_yy' : self.strain[2,:],
            'delta_sigma_xy_moy': self.delta_sigma_xy_moy,
            'nrj_sum_signal' : self.nrj_sum,
            'nrj_max_stft' : self.nrj_max}
        try:
            ch_dict['delta_sigma_xy_click'] = self.delta_sigma_xy_click
        except AttributeError :pass
        return ch_dict

    def convert_dict_to_ch(self,ch_dict):
        eps_xx = ch_dict['eps_xx']
        eps_yy = ch_dict['eps_yy']
        eps_xy = ch_dict['eps_xy']

        self.strain = np.array((eps_xx,eps_yy,eps_xy))
        self.delta_sigma_xy_moy = ch_dict['delta_sigma_xy_moy']
        self.nrj_sum = ch_dict['nrj_sum_signal']
        self.nrj_max = ch_dict['nrj_max_stft']

        try:
            self.delta_sigma_xy_click = ch_dict['delta_sigma_xy_click']
        except KeyError:pass
        pass

class event():
    def __init__(self, *args):
        arg = args[0]
        self.gages_position =  np.linspace(0,0.1425,20) ###### /!\

        if type(arg) == str:
            self.create_from_file(arg)
        elif type(arg) == dict:
            self.convert_dict_to_ev(arg)
        else: logging.error('Error: Argument type {} is not supported for event.__init__()'.format(type(arg)))


    def create_from_file(self,loc_event):
        event_dict,list_ch = facq.treat_event(loc_event)

        self.acoustic = event_dict['acoustic']
        self.trig = event_dict['trig']
        self.fs =event_dict['fs']
        self.fn = event_dict['fn']
        self.fs_gages = event_dict['fs_gages']
        self.fn_gages = event_dict['fn_gages']
        self.mu = event_dict['mu']

        self.gages = [channel(ch) for ch in list_ch]
        pass

    def click_event(self):
        rosette_pos = np.zeros(len(self.gages))
        eps_xx = np.zeros((len(self.gages),self.gages[0].strain.shape[1]))
        eps_xy,eps_yy = eps_xx.copy(),eps_xx.copy()
        for i,ch in enumerate(self.gages):
            rosette_pos[i] = self.gages_position[i]
            eps_xx[i,:],eps_xy[i,:],eps_yy[i,:] = ch.strain
        delta_sigma_xy_click,c_rupture = facq.click_stress_drop(eps_xx,eps_yy,eps_xy,rosette_pos)
        for i,ch in enumerate(self.gages):
            ch.delta_sigma_xy_click = delta_sigma_xy_click[i]
        self.c_rupture = c_rupture
        pass
    pass

    def convert_ev_to_dict(self):
        ev_dict = {
                'fn' : self.fn,
                'fs' : self.fs,
                'trig' : self.trig,
                'acoustic' : self.acoustic,
                'mu' : self.mu,
                'fn_gages': self.fn_gages,
                'fs_gages': self.fs_gages}

        for i,ch in enumerate(self.gages):
            ev_dict['rosette{}'.format(i+1)] = ch.convert_ch_to_dict()

        try:
            ev_dict['rupture_velocity'] = self.c_rupture
        except AttributeError:pass
        return ev_dict

    def convert_dict_to_ev(self,ev_dict):
        self.fn = ev_dict['fn']
        self.fs = ev_dict['fs']
        self.trig = ev_dict['trig']
        self.acoustic = ev_dict['acoustic']
        self.mu = ev_dict['mu']
        self.fn_gages = ev_dict['fn_gages']
        self.fs_gages = ev_dict['fs_gages']

        try:
            self.c_rupture=ev_dict['rupture_velocity']
        except KeyError:pass

        self.gages = []
        for key in ev_dict.keys():
            if key[:7] == 'rosette':
                self.gages.append(channel(ev_dict[key]))
        pass

        # def converted_peter_box(self):
        #     data = np.zeros((64,len(self.fn)))
        #     data[15,:] = self.fn
        #     data[31,:] = self.fs
        #     data[47,:] = self.trig
        #     data[63,:] = self.acoustic
        #
        #     for :
        #         self.gages[i].eps_xx,self.gages[i].eps_yy,self.gages[i].eps_xy =
        #     data[gages_ch] =

class slowmon():
    def __init__(self,*args):
        arg = args[0]
        if type(arg) == str:
            self.create_from_file(arg)
        elif type(arg) == dict:
            self.load_from_file(arg)
        else: logging.error('Error: Argument type {} is not supported for slowmon.__init__()'.format(type(arg)))
        pass

    def create_from_file(self, loc):
        gages_on_both_side = False
        data_zero = np.load(loc[:-11]+"event-001.npy",allow_pickle=True)
        data_sm = np.load(loc,allow_pickle=True)
        time_sm = np.load(loc[:-11]+"slowmon_time.npy")

        data_sm = np.transpose(np.transpose(data_sm)-np.mean(data_zero,axis=1)) # callibrate data set

        self.fs = daq.voltage_to_force(data_sm[31,:])
        self.fn = daq.voltage_to_force(data_sm[15,:])
        self.trig = data_sm[47,:]
        self.acoustic = data_sm[63,:]
        data_gages = data_sm[gages_ch]
        if gages_on_both_side : data_gages[1::2,:] = -data_gages[1::2,:]
        self.gages = facq.convert(data_gages)
        self.time = time_sm

        eps_yy = self.gages[1::3]
        eps_xy= self.gages[2::3]
        eps_xx = self.gages[0::3]

        if gages_on_both_side:
            eps_xx[1::2,:],eps_xy[1::2,:] =-eps_xy[1::2,:],-eps_yy[1::2,:]

        self.eps_yy = eps_yy
        self.eps_xy= eps_xy
        self.eps_xx = eps_xx
        self.mu = daq.smooth(self.fs,50) / daq.smooth(self.fn,50)
        pass

    def load_from_file(self,sm_dict):
        self.fs = sm_dict['fs']
        self.fn = sm_dict['fn']
        self.trig = sm_dict['trig']
        self.acoustic = sm_dict['acoustic']
        self.gages = sm_dict['gages']
        self.time = sm_dict['time']
        self.eps_yy = sm_dict['eps_yy']
        self.eps_xy= sm_dict['eps_xy']
        self.eps_xx = sm_dict['eps_xx']
        self.mu = sm_dict['mu']
        pass

    def convert_sm_to_dict(self):
        sm_dict = {
                'fn':self.fn,
                'fs':self.fs,
                'trig':self.trig,
                'acoustic':self.acoustic,
                'mu':self.mu,
                'gages':self.gages,
                'time':self.time,
                'eps_xx':self.eps_xx,
                'eps_yy':self.eps_yy,
                'eps_xy':self.eps_xy}
        return sm_dict


class manip():
    def __init__(self,*args):
        arg = args[0]
        if type(arg) == dict:
            self.load_from_file(arg)
        elif os.path.isfile(arg+'data_manip.npy'):
            man_dict = np.load(arg+'data_manip.npy',allow_pickle=True).all()
            self.load_from_file(man_dict)
        elif type(arg) == str:
            self.create_from_file(arg)
        else: logging.error('Error: Argument type {} is not supported for manip.__init__()'.format(type(arg)))
    def create_from_file(self,loc_manip):
        self.loc = loc_manip
        try:
            exec(daq.load_params(loc_manip+'parameters.txt'))
            print('\nFile {} has been found'.format(loc_manip+'parameters.txt'))
        except FileNotFoundError:
            logging.error("File not found : Try to add '/' at the end of the path")
            sys.exit()
        sm = slowmon(loc_manip+'slowmon.npy') # creation d'un objet slowmon pour contenir les donnees slowmon de la manip
        print(loc_manip)
        events = [event(loc_event) for loc_event in tqdm(glob.glob(loc_manip+'event-*.npy'))] # création d'autant d'objet evenements que nécessaire

        # self.gages_position = x
        self.sm = sm
        self.events = events

    def load_from_file(self,dict_man):
        self.sm = slowmon(dict_man['sm'])
        self.loc = dict_man['loc']
        self.events = []
        for key in dict_man.keys():
            if key[:6] == 'event-':
                self.events.append(event(dict_man[key]))

    def convert_man_to_dict(self):
        sm_dict = self.sm.convert_sm_to_dict()
        manip_dict = {  'sm':sm_dict,
                        'loc':self.loc}
        for i,ev in enumerate(self.events):
            manip_dict['event-{}'.format(i)] = ev.convert_ev_to_dict()
        return manip_dict

    def stat_acoustic(self):
        loc_manip = self.loc
        if not os.path.exists(loc_manip+'save_stats/'):
            from dev_stats_detection_multithreading import main_traitement
            loc_tdms_data = glob.glob(loc_manip+'*.tdms')
            print('tdms file found : {}'.format(loc_tdms_data))
            main_traitement(loc_manip,loc_tdms_data[0][len(loc_manip):])

        self.stats_pos_idx = np.load(loc_manip+'save_stats/pos_idx.npy',allow_pickle=True)
        self.stats_neg_idx = np.load(loc_manip+'save_stats/neg_idx.npy',allow_pickle=True)
        self.stats_peak_idx = np.load(loc_manip+'save_stats/peak_idx.npy',allow_pickle=True)
        self.stats_nrj_sum_signal = np.load(loc_manip+'save_stats/nrj_sum_signal.npy',allow_pickle=True)
        self.stats_nrj_sum_stft = np.load(loc_manip+'save_stats/nrj_sum_stft.npy',allow_pickle=True)
        self.stats_nrj_max_stft = np.load(loc_manip+'save_stats/nrj_max_stft.npy',allow_pickle=True)
        self.stats_dt = np.load(loc_manip+'save_stats/dt.npy',allow_pickle=True)
    def save(self,man_dict):
        man_dict = self.convert_man_to_dict
        np.save(self.loc+'data_manip',man_dict)





class catalog():
    '''
    Objet qui contient plusieurs manip, qui peut servir à rassembler un maximum de statistique.
    2 modes de création :

    1) Si la création du dictionnaire n'a pas été faite, création à partir des données PeterBox.
    L'arborescence attendue est :
    --loc
        (--meta_data.txt)
        --Manip1    # fichier de sortie de la PeterBox (après demux_all_files) éventuellement compléter par des fichiers de données acoustiques.
            # Données PeterBox
            --parameters.txt
            --slowmon.npy
            --slowmon_time.npy
            --event-001.npy
            ...
            --event-N.npy
            # Données acoustiques (optionnelles)
            (--Sampling.tdms)
            (--save_stats # fichier créé en utilisant le programme 'dev_stats_detection_multithreading.py'
                --pos_idx.npy
                --neg_idx.npy
                --nrj_sum_signal.npy
                --nrj_sum_stft.npy
                --nrj_max_stft.npy
                --dt )
        --Manip2
            ...
        ...
        --ManipN

        2) A partir d'un dictionnaire enregistré via np.save().
        La structure du dictionnaire est :



        En sortie, on créer un objet catalog dont la structure est :

        cat
            (.meta_data)
            .manips[0]
                (.meta_data)
                .sm
                    .
                .event-001
                ...
                .event-N
            ...
            .manips[N]
    '''
    def __init__(self, loc):
        '''
        Le mode de création de l'objet catalogue est choisi en fonction de l'existance du fichier de sauvegarde de dictionnaire (loc+'data_catalog.npy').
        '''
        if os.path.isfile(loc+'data_catalog.npy'):
            self.load_from_file(loc+'data_catalog.npy')
        else : self.create_from_file(loc)

    def load_from_file(self,loc):
        '''
        Mode de création à partir d'un fichier dict déjà existant (mode 2)
        '''
        cat_dict = np.load(loc,allow_pickle=True).all()
        self.convert_dict_to_cat(cat_dict)

    def create_from_file(self,loc):
        '''
        Mode de création à partir des données brutes (mode 1)
        '''
        # self.meta_data = np.loadtxt(loc+ 'meta_data.txt')
        self.manips = [manip(loc_manip) for loc_manip in glob.glob(loc+'*/')]

    def convert_cat_to_dict(self):
        '''
        Méthode utilisée dans la procédure d'enregistrement du catalogue sur le disque.
        Chaque niveau (catalogue -> manip -> slowmon + event -> channel) appelle la méthode 'convert_*_to_dict' du niveau inférieur (jusqu'au niveau channel) et réunie tout son contenu dans un dictionnaire.
        '''
        cat_dict = {}
        for i,manip in enumerate(self.manips):
            cat_dict[i] = manip.convert_man_to_dict()
        return cat_dict

    def save(self,loc):
        cat_dict = self.convert_cat_to_dict()
        np.save(loc+'data_catalog',cat_dict)

    def convert_dict_to_cat(self,cat_dict):
        '''
        Méthode utilisée dans la procédure de lecture d'un catalogue enregistré sur le disque sous forme de dictionnaire.npy.
        De la même façon que pour 'convert_cat_to_dict', chaque niveau appelle le niveau inférieur pour récupérer les données des catalogues en objets cat/manip/event...
        '''
        self.manips = []
        for i in cat_dict.keys():
            self.manips.append(manip(cat_dict[i]))
        # self.meta_data = np.loadtxt(loc+ 'meta_data.txt')
