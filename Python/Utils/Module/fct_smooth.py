# -*- coding: utf-8 -*-

import numpy as np
import tsmoothie
from numpy.fft import fft, fftfreq
from tqdm import tqdm

import scipy.signal as sg

# ------------------------------------------
def smooth(sig, win_len = 30):
    ''' Just a nice, quick and easy to use smoother based on tsmoothie'''
    smoother = tsmoothie.ConvolutionSmoother(win_len, 'ones')
    return smoother.smooth(sig).smooth_data[0]


# ------------------------------------------
def tsmooth(data, wt, acq_fr: int = 1):
    """Function to extract one single data batch

    :param b: number of batch to extract
    :return: liste of single data and informations associated: f, t, t_V, batch_ini, batch name
    """

    win_len = int(wt * acq_fr)
    smoother = tsmoothie.ConvolutionSmoother(win_len, 'ones')

    return smoother.smooth(data).smooth_data[0]


# ------------------------------------------
def tsmooth_low_up(data, wt, acq_fr: int = 1):
    """Function to extract one single data batch

    :param b: number of batch to extract
    :return: liste of single data and informations associated: f, t, t_V, batch_ini, batch name
    """

    win_len = int(wt * acq_fr)
    smoother = tsmoothie.ConvolutionSmoother(win_len, 'ones')
    smoother.smooth(data)

    low, up = smoother.get_intervals('sigma_interval')

    return smoother.smooth_data[0], low[0], up[0]
