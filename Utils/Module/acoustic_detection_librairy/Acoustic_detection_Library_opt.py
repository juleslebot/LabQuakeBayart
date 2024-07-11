"""
Created on Tue Oct 24 11:11:23 2023

@author: Ptashanna THIRAUX, Stephane ROUX, Adèle DOUIN, Osvanny RAMOS.
"""

import scipy
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.decomposition import NMF, PCA

from Utils.Module.acoustic_detection_librairy.custom_stft_nworkers import my_stft

def Gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def finding_thresh(STFT, nbins=1000, plot=False):
    p, x = np.histogram(STFT, bins=nbins)

    internal_thresh = 1
    ecartold = 0

    while (True):

        p2 = p[p > internal_thresh]

        centers = (x[1:] + x[0:-1]) / 2
        centers = centers[p > internal_thresh]

        p2 = p2 / np.sum(p2) / np.mean(np.diff(centers))  # ou np.diff(centers) ?
        std = np.std(STFT)
        mean = np.mean(STFT)
        a = 1 / np.sqrt(2 * np.pi * std ** 2)
        # p2 = scipy.signal.savgol_filter(p2,100,3)

        # pop,pcov = curve_fit(gaussfunction, centers, p2,p0 = [mean,std])
        pop, pcov = curve_fit(Gauss, centers, p2, p0=[a, mean, std])

        # gauss_estimated = gaussfunction(centers,pop[0],pop[1])
        gauss_estimated = Gauss(centers, pop[0], pop[1], pop[2])
        ecart = np.sum(p2 - gauss_estimated) ** 2

        if np.abs(ecart - ecartold) < 1e-6:

            xtmp = x[x > centers[-1]]
            Final_threshold = xtmp[0]

            if plot:
                plt.figure()
                plt.clf()
                plt.plot(centers, np.log(p2), 'k', label="data")
                plt.plot(centers, np.log(Gauss(centers, pop[0], pop[1], pop[2])), "b",
                         label="fit err : " + str(np.round(ecart, 4)))
                plt.legend()

            return Final_threshold, ecart

        else:

            ecartold = ecart
            internal_thresh = internal_thresh + 1


def Find_peaks(data_process, threshold=0.1, fs=10 ** 5):
    """Given a processed data by STFT or CWT, this function finds the beginning and the end of events"""
    # data_process =(data_process-np.min(data_process))/(np.max(data_process)-np.min(data_process)) #Normalisation

    peaks, properties = scipy.signal.find_peaks(data_process, height=threshold,
                                                width=0)  # I'm finding the peaks in processed data

    peak_heights = properties['peak_heights']

    # plt.figure()
    # plt.plot(data_process)

    data_process_filter = np.copy(data_process)

    data_process_filter[np.where(data_process_filter < np.min(peak_heights))] = 0

    # plt.figure()
    # plt.plot(data_process_filter)

    peaks_final, properties_final = scipy.signal.find_peaks(data_process_filter, width=0, rel_height=1)

    index_beg = properties_final['left_bases']
    index_end = properties_final['right_bases']

    # event_duration = (properties_final['right_ips'] - properties_final['left_ips'])/fs

    return index_beg, index_end, peaks_final, properties_final


def Acoustic_detection_STFT_only(data, nseg, nover, nbin, fs=10 ** 5, nworkers=None, plot=False):
    """STFT computation"""
    f, t, Zxx = my_stft(data, fs, nperseg=nseg, noverlap=nover, nworkers=nworkers)
    STFT = np.sum(np.abs(Zxx), axis=0)

    """Finding the threshold to cut down the noise in STFT by gaussian fitting"""
    threshold, ecart = finding_thresh(STFT, nbins=nbin, plot=plot)

    """Finding the peaks in STFT"""
    index_beg, index_end, peaks_final, properties_final = Find_peaks(STFT, threshold=threshold)
    Zxx = 0

    return threshold, index_beg, index_end, STFT


def Acoustic_detection_STFT_NMF(data, nseg, nover, nbin, fs=10 ** 5, nworkers=None, plot=False):
    """STFT computation"""
    f, t, Zxx = my_stft(data, fs, nperseg=nseg, noverlap=nover, nworkers=nworkers)
    tmp = np.abs(Zxx)

    model = NMF(n_components=2, max_iter=500)
    W = model.fit_transform(np.log1p(tmp))
    H = model.components_
    """Finding the threshold to cut down the noise in STFT by gaussian fitting"""

    threshold, ecart = finding_thresh(H[1, :], nbins=nbin, plot=plot)

    """Finding the peaks in STFT"""

    index_beg, index_end, peaks_final, properties_final = Find_peaks(H[1, :], threshold=threshold)

    return W[:, 1], H[1, :], threshold, index_beg, index_end, ecart


def Acoustic_detection_STFT_PCA(data, nseg, nover, nbin, fs=10 ** 5, nworkers=None, plot=False):
    """STFT computation"""
    f, t, Zxx = my_stft(data, fs, nperseg=nseg, noverlap=nover, nworkers=nworkers)
    tmp = np.abs(Zxx)

    model = PCA(n_components=2)
    H = model.fit_transform(np.log1p(tmp))
    W = model.components_

    print(len(W[1, :]))
    """Finding the threshold to cut down the noise in STFT by gaussian fitting"""

    # threshold, ecart = finding_thresh(W[1, :], nbins=nbin, plot=plot)
    threshold, ecart = 0, 0

    """Finding the peaks in STFT"""

    index_beg, index_end, peaks_final, properties_final = Find_peaks(W[1, :], threshold=threshold)

    return W[1, :], H[:, 1], threshold, index_beg, index_end, ecart

