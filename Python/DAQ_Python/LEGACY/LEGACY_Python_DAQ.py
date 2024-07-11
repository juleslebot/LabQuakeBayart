"""
Python Functions to use a DAQ
"""

## Imports
# Science
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy import interpolate
# DAQ
import nidaqmx
from nidaqmx.stream_readers import AnalogMultiChannelReader
from nidaqmx import constants
# random
import threading
import pickle
import time
import os

## Definitions

def calibration_functions(max_D63_1=3.43,max_D63_2=1.77):
    """
    The calibration of the force detectors and distance detectors
    Parameters :
        max_D63_1 and 2 (float): The maximum of both the D63 captors
    Output : the 4 calibration functions
    """
    # First the D63 position detector
    D63_cal = np.genfromtxt('D:/Users/Manips/Documents/DATA/FRICS/calibration_distance_detector_D63.txt', delimiter=';')
    D63_cal[0,0]=0. # is nan
    D63_cal[:,1]=D63_cal[:,1]/np.max(D63_cal[:,1]) # normalize
    D63_cal[-2,0]=-1. # mark as -1 when out of bound
    D63_cal[-1,1]=-10. # define the "-1" zone as larger to avoid "out of range" errors
    D63_cal[-1,0]=-1.
    max_index =np.argmax(D63_cal[:,1]) #find the max to only keep the long range part
    cal_func_D63_1=interpolate.interp1d(max_D63_1*D63_cal[max_index:,1],D63_cal[max_index:,0])
    cal_func_D63_2=interpolate.interp1d(max_D63_2*D63_cal[max_index:,1],D63_cal[max_index:,0])

    #Then the two force detectors
    k=5e4
    a1,b1,offset1=490,0.21,-26.4 # check that
    a2,b2,offset2=-495,-0.2,-22.4 # check that
    def cal_func_F(x,a,b,offset):
        return((x-b)/a*k-offset)

    def cal_func_Fn(x):
        return(cal_func_F(x,a1,b1,offset1))
    def cal_func_Fs(x):
        return(cal_func_F(x,a2,b2,offset2))
    return(cal_func_Fn,cal_func_Fs,cal_func_D63_1,cal_func_D63_2)

def calibration_functions_2():
    """
    The calibration of the force detectors and one uncalibrated TTL :
    Output : the 3 calibration functions
    """
    # the two force detectors
    k=5e4
    a1,b1,offset1=490,0.21,-26.4 # check that
    a2,b2,offset2=-495,-0.2,-22.4 # check that
    def cal_func_F(x,a,b,offset):
        return((x-b)/a*k-offset)

    def cal_func_Fn(x):
        return(cal_func_F(x,a1,b1,offset1))
    def cal_func_Fs(x):
        return(cal_func_F(x,a2,b2,offset2))

    def cal_volt(x):
        return(x)
    return(cal_func_Fn,cal_func_Fs,cal_volt)

def calibration_functions_force():
    """
    The calibration of the force detectors and one uncalibrated TTL :
    Output : the 3 calibration functions
    """
    # the two force detectors
    # F = G * V
    G=500/3

    def cal_func_Fn(x):
        return(G*x)
    def cal_func_Fs(x):
        return(G*x)

    # simple identity function
    def cal_volt(x):
        return(x)
    return(cal_func_Fn,cal_func_Fs,cal_volt)


def calibration_function_D63(max_D63=3.43):
    """
    The calibration of a single D63 :
    Output : the calibration functions
    """
    # the D63 position detector
    D63_cal = np.genfromtxt('D:/Users/Manips/Documents/DATA/FRICS/calibration_distance_detector_D63.txt', delimiter=';')
    D63_cal[0,0]=0. # is nan
    D63_cal[:,1]=D63_cal[:,1]/np.max(D63_cal[:,1]) # normalize
    D63_cal[-2,0]=-1. # mark as -1 when out of bound
    D63_cal[-1,1]=-10. # define the "-1" zone as larger to avoid "out of range" errors
    D63_cal[-1,0]=-1.
    max_index =np.argmax(D63_cal[:,1]) #find the max to only keep the long range part
    cal_func_D63=interpolate.interp1d(max_D63*D63_cal[max_index:,1],D63_cal[max_index:,0])
    return(cal_func_D63)

def calibration_function_D63_short_range(max_D63=3.43):
    """
    The calibration of a single D63 :
    Output : the calibration functions
    """
    # the D63 position detector
    D63_cal = np.genfromtxt('D:/Users/Manips/Documents/DATA/FRICS/calibration_distance_detector_D63.txt', delimiter=';')
    D63_cal[0,0]=0. # is nan
    D63_cal[:,1]=D63_cal[:,1]/np.max(D63_cal[:,1]) # normalize
    #D63_cal[-2,0]=-1. # mark as -1 when out of bound
    #D63_cal[-1,1]=-10. # define the "-1" zone as larger to avoid "out of range" errors
    #D63_cal[-1,0]=-1.
    max_index =np.argmax(D63_cal[:,1]) #find the max to only keep the short range part
    D63_cal=D63_cal[:max_index+2]
    D63_cal[-2,1]=-np.infty
    D63_cal[-1,1]=-np.infty
    cal_func_D63=interpolate.interp1d(max_D63*D63_cal[:,1],D63_cal[:,0])
    return(cal_func_D63)

def cfg_read_task(acquisition,chans_in,sampling_freq_in,buffer_in_size,dev="Dev1"):
    """
    Sets up the reading task with the chans_in first channels
    Parameters:
        acquisition: nidaqmx.Task() object
        chans_in          (int): number of channels
        sampling_freq_in  (int): sampling frequency in Hz
        buffer_in_size    (int): size of the data buffer
    """
    acquisition.ai_channels.add_ai_voltage_chan("{}/ai0:{}".format(dev,chans_in-1))
    acquisition.timing.cfg_samp_clk_timing(rate=sampling_freq_in, sample_mode=constants.AcquisitionType.CONTINUOUS,
                                        samps_per_chan=buffer_in_size)
    return(None)

def refresh(DAQ_data, ax, chans_in, sampling_freq_in, time_window, xlabel="time (s)", ylabels="voltage (V)",navg_roll=1,navg=1):
    """
    Refreshes a live plot
    Parameters:
        DAQ_data          (arr): data to plot
        ax                (ax) : ax to refresh
        chans_in          (int): number of channels
        sampling_freq_in  (int): sampling frequency in Hz
        time_window       (int): lenght of display in second
        navg_roll         (int): size of the rolling average convolution (display only)
        navg              (int): size of the real average (directly on measure)
    """
    # Manage y labels
    if not isinstance(ylabels, list):
        ylabels=[ylabels for _ in range(chans_in)]
    elif len(ylabels)!=chans_in:
        raise Exception("You have {} labels, expected {}".format(len(ylabels),chans_in))

    # plot curves
    for i in range(chans_in):
        ax[i].clear()
        ax[i].plot(rolling_average(DAQ_data[i, -int(sampling_freq_in * time_window):].T,navg_roll)) # Rolling window
        ax[i].set_ylabel(ylabels[i])
        ax[i].yaxis.set_ticks_position('both')
        ax[i].grid(True)

    # Label and axis formatting
    ax[-1].set_xlabel(xlabel)
    xticks = np.arange(0, DAQ_data[0, -int(sampling_freq_in * time_window):].size, sampling_freq_in/navg)
    xticklabels = np.arange(0, xticks.size, 1)
    ax[-1].set_xticks(xticks)
    ax[-1].set_xticklabels(xticklabels)
    return(None)


def live_plot(chans_in, sampling_freq_in = 1000, refresh_rate_plot = 30, time_window=5,
                 save=False, buffer_in_size = None, calibration_functions=None, ylabels="voltage (V)",relative=[],navg_roll=1,navg=1,dev="Dev1",sharey='none',no_plot=False,**kwargs):
    """
    Makes a live plot of the chans_in first channels. Press Enter to stop the acquisition
    Parameters:
        chans_in          (int): number of channels
        sampling_freq_in  (int): sampling frequency in Hz
        refresh_rate_plot (int): plotting refresh rate (might be capped by CPU usage)
        save              (Boo): If True, saves the data when stopping the acquisition
        buffer_in_size    (int): size of the data buffer (specify a value if you encounter an error)
                                 This value has to either divide or be divided by navg
                                 it corresponds to the number of samples acquired simulteniously
        time_window       (int): lenght of display in second
        calibration_functions  : a functions that outputs a list of chans_in functions for calibration
        ylabels           (str): a string for the y label of the plot. Can be a list of strings.
        relative          (lst): a list of ints such that all channels in the list will be plotted relatively to their first value
        navg_roll         (int): size of the rolling average convolution (display only)
        navg              (int): size of the real average (directly on measure)
    """
    time_window /= navg
    # Define a buffer size if none is specified
    if not buffer_in_size:
        buffer_in_size=sampling_freq_in//10

    # Initialize data placeholders
    buffer_in = np.zeros((chans_in, buffer_in_size))
    global DAQ_data
    DAQ_data = np.zeros((chans_in, 1))  # will contain a first column with zeros but that's dealt with line 157
    global First_loop
    First_loop=True

    if calibration_functions:
        functions=calibration_functions(**kwargs)

        def calibration(a):
            if chans_in==1:
                a[0]=functions(a[0])
            else:
                for i in range(chans_in):
                    a[i]=functions[i](a[i])
            return(a)
    else:
        def calibration(a):
            return(a)

    # Definitions of basic functions
    def ask_user():
        """The "Press enter to stop the acquisition" part"""
        global running
        input("Press ENTER/RETURN to stop acquisition.")
        running = False
        return(None)

    def reading_task_callback(task_idx, event_type, num_samples, callback_data):
        """
        Intermediary function that reads the data from the DAQ
        """
        # the global way is really dirty, but it works fine
        global DAQ_data
        global First_loop
        global relat
        if running:
            # It may be wiser to read slightly more than num_samples here, to make sure one does not miss any sample,
            # see: https://documentation.help/NI-DAQmx-Key-Concepts/contCAcqGen.html
            stream_in.read_many_sample(buffer_in, num_samples, timeout=constants.WAIT_INFINITELY)
            new_data=buffer_in
            a,b=new_data.shape
            new_data=new_data[:,:navg*(b//navg)].reshape(a,b//navg,navg)
            new_data=np.median(new_data,axis=-1)
            new_data = calibration(new_data)
            if First_loop:
                DAQ_data = np.append(DAQ_data, new_data, axis=1)  # appends buffered data to total variable DAQ_data
                DAQ_data=DAQ_data[:,1:] # dealing with the first column
                relat = np.average(DAQ_data,axis=1).reshape(len(DAQ_data),1)
                if relative!=[]:
                    DAQ_data[relative]-=relat[relative]
                First_loop=False
            else:
                if relative!=[]:
                    new_data[relative]-=relat[relative]
                DAQ_data = np.append(DAQ_data, new_data, axis=1)
        return(0)

    # Configure and setup the tasks
    task_in = nidaqmx.Task()
    cfg_read_task(task_in,chans_in,sampling_freq_in,buffer_in_size,dev=dev)
    stream_in = AnalogMultiChannelReader(task_in.in_stream)
    task_in.register_every_n_samples_acquired_into_buffer_event(buffer_in_size, reading_task_callback)

    # Start threading to prompt user to stop
    thread_user = threading.Thread(target=ask_user)
    thread_user.start()

    # Main loop
    global running
    running = True
    task_in.start()

    if not no_plot:
        # Plotting
        if chans_in==1 :
            f, ax = plt.subplots(chans_in, 1)
            ax=[ax]
            while running:
                refresh(DAQ_data, ax, chans_in, sampling_freq_in, time_window, xlabel="time (s)", ylabels=ylabels,navg_roll=navg_roll,navg=navg)
                if not save:
                    DAQ_data=DAQ_data[:,-int(sampling_freq_in * time_window):]
                plt.pause(1/refresh_rate_plot)  # required for dynamic plot to work (if too low, nulling performance bad)

        else:
            f, ax = plt.subplots(chans_in, 1, sharex='all', sharey=sharey)
            while running:
                refresh(DAQ_data, ax, chans_in, sampling_freq_in, time_window, xlabel="time (s)", ylabels=ylabels,navg_roll=navg_roll,navg=navg)
                if not save:
                    DAQ_data=DAQ_data[:,-int(sampling_freq_in * time_window):]
                plt.pause(1/refresh_rate_plot)  # required for dynamic plot to work (if too low, nulling performance bad)
    else:
        while running:
            time.sleep(1)

    # Close task to clear connection once done
    task_in.close()

    plt.close('all')

    if save:
        if save == True:
            save_data_visual(DAQ_data[:,1:])
        elif save[-4:]==".txt":
            np.savetxt(save,DAQ_data[:,1:])
        else:
            np.save(save,DAQ_data[:,1:])
    return(DAQ_data)






# Dummy function that is used in order to have a double daq acquisition, with two separate devices.
def live_plot2(chans_in, sampling_freq_in = 1000, refresh_rate_plot = 30, time_window=5,
                 save=False, buffer_in_size = None, calibration_functions=None, ylabels="voltage (V)",relative=[],navg_roll=1,navg=1,dev="Dev1",sharey='none',no_plot=False,**kwargs):
    """
    Makes a live plot of the chans_in first channels. Press Enter to stop the acquisition
    Parameters:
        chans_in          (int): number of channels
        sampling_freq_in  (int): sampling frequency in Hz
        refresh_rate_plot (int): plotting refresh rate (might be capped by CPU usage)
        save              (Boo): If True, saves the data when stopping the acquisition
        buffer_in_size    (int): size of the data buffer (specify a value if you encounter an error)
                                 This value has to either divide or be divided by navg
                                 it corresponds to the number of samples acquired simulteniously
        time_window       (int): lenght of display in second
        calibration_functions  : a functions that outputs a list of chans_in functions for calibration
        ylabels           (str): a string for the y label of the plot. Can be a list of strings.
        relative          (lst): a list of ints such that all channels in the list will be plotted relatively to their first value
        navg_roll         (int): size of the rolling average convolution (display only)
        navg              (int): size of the real average (directly on measure)
    """
    time_window /= navg
    # Define a buffer size if none is specified
    if not buffer_in_size:
        buffer_in_size=sampling_freq_in//10

    # Initialize data placeholders
    buffer_in = np.zeros((chans_in, buffer_in_size))
    global DAQ_data2
    DAQ_data2 = np.zeros((chans_in, 1))  # will contain a first column with zeros but that's dealt with line 157
    global First_loop
    First_loop=True

    if calibration_functions:
        functions=calibration_functions(**kwargs)

        def calibration(a):
            if chans_in==1:
                a[0]=functions(a[0])
            else:
                for i in range(chans_in):
                    a[i]=functions[i](a[i])
            return(a)
    else:
        def calibration(a):
            return(a)

    # Definitions of basic functions
    def ask_user():
        """The "Press enter to stop the acquisition" part"""
        global running
        input("Press ENTER/RETURN to stop acquisition.")
        running = False
        return(None)

    def reading_task_callback(task_idx, event_type, num_samples, callback_data):
        """
        Intermediary function that reads the data from the DAQ
        """
        # the global way is really dirty, but it works fine
        global DAQ_data2
        global First_loop
        global relat
        if running:
            # It may be wiser to read slightly more than num_samples here, to make sure one does not miss any sample,
            # see: https://documentation.help/NI-DAQmx-Key-Concepts/contCAcqGen.html
            stream_in.read_many_sample(buffer_in, num_samples, timeout=constants.WAIT_INFINITELY)
            new_data=buffer_in
            a,b=new_data.shape
            new_data=new_data[:,:navg*(b//navg)].reshape(a,b//navg,navg)
            new_data=np.median(new_data,axis=-1)
            new_data = calibration(new_data)
            if First_loop:
                DAQ_data2 = np.append(DAQ_data2, new_data, axis=1)  # appends buffered data to total variable DAQ_data2
                DAQ_data2=DAQ_data2[:,1:] # dealing with the first column
                relat = np.average(DAQ_data2,axis=1).reshape(len(DAQ_data2),1)
                if relative!=[]:
                    DAQ_data2[relative]-=relat[relative]
                First_loop=False
            else:
                if relative!=[]:
                    new_data[relative]-=relat[relative]
                DAQ_data2 = np.append(DAQ_data2, new_data, axis=1)
        return(0)

    # Configure and setup the tasks
    task_in = nidaqmx.Task()
    cfg_read_task(task_in,chans_in,sampling_freq_in,buffer_in_size,dev=dev)
    stream_in = AnalogMultiChannelReader(task_in.in_stream)
    task_in.register_every_n_samples_acquired_into_buffer_event(buffer_in_size, reading_task_callback)

    # Start threading to prompt user to stop
    thread_user = threading.Thread(target=ask_user)
    thread_user.start()

    # Main loop
    global running
    running = True
    task_in.start()

    if not no_plot:
        # Plotting
        if chans_in==1 :
            f, ax = plt.subplots(chans_in, 1)
            ax=[ax]
            while running:
                refresh(DAQ_data2, ax, chans_in, sampling_freq_in, time_window, xlabel="time (s)", ylabels=ylabels,navg_roll=navg_roll,navg=navg)
                if not save:
                    DAQ_data2=DAQ_data2[:,-int(sampling_freq_in * time_window):]
                plt.pause(1/refresh_rate_plot)  # required for dynamic plot to work (if too low, nulling performance bad)

        else:
            f, ax = plt.subplots(chans_in, 1, sharex='all', sharey=sharey)
            while running:
                refresh(DAQ_data2, ax, chans_in, sampling_freq_in, time_window, xlabel="time (s)", ylabels=ylabels,navg_roll=navg_roll,navg=navg)
                if not save:
                    DAQ_data2=DAQ_data2[:,-int(sampling_freq_in * time_window):]
                plt.pause(1/refresh_rate_plot)  # required for dynamic plot to work (if too low, nulling performance bad)
    else:
        while running:
            time.sleep(1)

    # Close task to clear connection once done
    task_in.close()

    plt.close('all')

    if save:
        if save == True:
            save_data_visual(DAQ_data2[:,1:])
        elif save[-4:]==".txt":
            np.savetxt(save,DAQ_data2[:,1:])
        else:
            np.save(save,DAQ_data2[:,1:])
    return(DAQ_data2)








def save_data_visual(data):
    """
    Just a quick function to save an array as either a txt or a npy file, with a visual interface
    """
    import tkinter
    from tkinter.filedialog import asksaveasfilename
    files = [('Text Document', '*.txt'),('Numpy Files', '*.npy')]
    root = tkinter.Tk()
    root.update()
    file = asksaveasfilename(filetypes = files, defaultextension = files)
    root.destroy()
    if file == '':
        print("File not saved")
    elif file[-1]=="t":
        np.savetxt(file,data)
    else:
        np.savetxt(file,data)
    return(None)

def quick_plot_data(data, sampling_freq_in, xlabel="Time (s)", ylabels="voltage (V)"):
    """
    makes a quick plot
    Parameters:
        data              (arr): data to plot
        sampling_freq_in  (int): sampling frequency in Hz
        time_window       (int): lenght of display in second
    """
    chans_in=len(data)
    f, ax = plt.subplots(chans_in, 1, sharex='all', sharey='none')
    # Manage y labels
    if not isinstance(ylabels, list):
        ylabels=[ylabels for _ in range(chans_in)]
    elif len(ylabels)!=chans_in:
        raise Exception("You have {} labels, expected {}".format(len(ylabels),chans_in))
    x=np.arange(data[0,:].size)/sampling_freq_in
    # plot curves
    for i in range(chans_in):
        ax[i].plot(x,data[i,:])
        ax[i].set_ylabel(ylabels[i])
        ax[i].grid(which="both")


    # Label and axis formatting
    ax[-1].set_xlabel(xlabel)
    return(f)

