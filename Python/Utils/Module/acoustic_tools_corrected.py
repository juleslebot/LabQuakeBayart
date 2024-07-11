# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 14:54:03 2019

@author: Victor
"""
from nptdms import TdmsFile
import numpy as np
from matplotlib import pyplot as plt


def open_tdms(path):
    """ Open the tdms file and store each of the 6 signals in a dictionnary. """
    file = TdmsFile(path)
    a = file.as_dataframe()
    signals = {}
    signals[0] = np.array(a[a.columns[0]])
    signals[1] = np.array(a[a.columns[1]])
    signals[2] = np.array(a[a.columns[2]])
    signals[3] = np.array(a[a.columns[3]])
    signals[4] = np.array(a[a.columns[4]])
    signals[5] = np.array(a[a.columns[5]])
    signals[6] = np.array(a[a.columns[6]])

    # for k in range(8):
    #    if k == 2:continue
    #   plt.plot(signals[k][::50]+15000*k)
    # plt.show()
    # force = np.array(a[a.columns[6]])
    return signals


def open_tdms_f(path):
    file = TdmsFile(path)
    a = file.as_dataframe()
    signals = {}
    signals[0] = np.array(a[a.columns[0]])
    signals[1] = np.array(a[a.columns[1]])
    return signals


def open_tdms_jules_acoustic(path):
    file = TdmsFile(path)
    a = file.as_dataframe()
    signals = {}
    signals[0] = np.array(a[a.columns[0]])
    signals[1] = np.array(a[a.columns[1]])
    signals[2] = np.array(a[a.columns[2]])
    return signals
