# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 11:28:00 2023

@author: Ptashanna THIRAUX, Stephane G. ROUX
"""

import numpy as np
from scipy import fft as sp_fft
import warnings
import operator
from scipy.signal.windows import get_window
from scipy.signal._arraytools import const_ext, even_ext, odd_ext, zero_ext
from scipy.signal._spectral_py import _triage_segments
import scipy.signal._signaltools as _signaltools


def my_stft(x, fs=1.0, window='hann', nperseg=256, noverlap=None, nfft=None,
            detrend=False, return_onesided=True, boundary='zeros', padded=True,
            axis=-1, scaling='spectrum', nworkers=None):
    r"""Compute the Short Time Fourier Transform (STFT).

    STFTs can be used as a way of quantifying the change of a
    nonstationary signal's frequency and phase content over time.

    Parameters
    ----------
    x : array_like
        Time series of measurement values
    fs : float, optional
        Sampling frequency of the `x` time series. Defaults to 1.0.
    window : str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg. Defaults
        to a Hann window.
    nperseg : int, optional
        Length of each segment. Defaults to 256.
    noverlap : int, optional
        Number of points to overlap between segments. If `None`,
        ``noverlap = nperseg // 2``. Defaults to `None`. When
        specified, the COLA constraint must be met (see Notes_v01.md below).
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired. If
        `None`, the FFT length is `nperseg`. Defaults to `None`.
    detrend : str or function or `False`, optional
        Specifies how to detrend each segment. If `detrend` is a
        string, it is passed as the `type` argument to the `detrend`
        function. If it is a function, it takes a segment and returns a
        detrended segment. If `detrend` is `False`, no detrending is
        done. Defaults to `False`.
    return_onesided : bool, optional
        If `True`, return a one-sided spectrum for real data. If
        `False` return a two-sided spectrum. Defaults to `True`, but for
        complex data, a two-sided spectrum is always returned.
    boundary : str or None, optional
        Specifies whether the input signal is extended at both ends, and
        how to generate the new values, in order to center the first
        windowed segment on the first input point. This has the benefit
        of enabling reconstruction of the first input point when the
        employed window function starts at zero. Valid options are
        ``['even', 'odd', 'constant', 'zeros', None]``. Defaults to
        'zeros', for zero padding extension. I.e. ``[1, 2, 3, 4]`` is
        extended to ``[0, 1, 2, 3, 4, 0]`` for ``nperseg=3``.
    padded : bool, optional
        Specifies whether the input signal is zero-padded at the end to
        make the signal fit exactly into an integer number of window
        segments, so that all of the signal is included in the output.
        Defaults to `True`. Padding occurs after boundary extension, if
        `boundary` is not `None`, and `padded` is `True`, as is the
        default.
    axis : int, optional
        Axis along which the STFT is computed; the default is over the
        last axis (i.e. ``axis=-1``).
    scaling: {'spectrum', 'psd'}
        The default 'spectrum' scaling allows each frequency line of `Zxx` to
        be interpreted as a magnitude spectrum. The 'psd' option scales each
        line to a power spectral density - it allows to calculate the signal's
        energy by numerically integrating over ``abs(Zxx)**2``.

        .. versionadded:: 1.9.0

    Returns
    -------
    f : ndarray
        Array of sample frequencies.
    t : ndarray
        Array of segment times.
    Zxx : ndarray
        STFT of `x`. By default, the last axis of `Zxx` corresponds
        to the segment times.

    See Also
    --------
    istft: Inverse Short Time Fourier Transform
    check_COLA: Check whether the Constant OverLap Add (COLA) constraint
                is met
    check_NOLA: Check whether the Nonzero Overlap Add (NOLA) constraint is met
    welch: Power spectral density by Welch's method.
    spectrogram: Spectrogram by Welch's method.
    csd: Cross spectral density by Welch's method.
    lombscargle: Lomb-Scargle periodogram for unevenly sampled data

    Notes_v01.md
    -----
    In order to enable inversion of an STFT via the inverse STFT in
    `istft`, the signal windowing must obey the constraint of "Nonzero
    OverLap Add" (NOLA), and the input signal must have complete
    windowing coverage (i.e. ``(x.shape[axis] - nperseg) %
    (nperseg-noverlap) == 0``). The `padded` argument may be used to
    accomplish this.

    Given a time-domain signal :math:`x[n]`, a window :math:`w[n]`, and a hop
    size :math:`H` = `nperseg - noverlap`, the windowed frame at time index
    :math:`t` is given by

    .. math:: x_{t}[n]=x[n]w[n-tH]

    The overlap-add (OLA) reconstruction equation is given by

    .. math:: x[n]=\frac{\sum_{t}x_{t}[n]w[n-tH]}{\sum_{t}w^{2}[n-tH]}

    The NOLA constraint ensures that every normalization term that appears
    in the denomimator of the OLA reconstruction equation is nonzero. Whether a
    choice of `window`, `nperseg`, and `noverlap` satisfy this constraint can
    be tested with `check_NOLA`.

    .. versionadded:: 0.19.0

    References
    ----------
    .. [1] Oppenheim, Alan V., Ronald W. Schafer, John R. Buck
           "Discrete-Time Signal Processing", Prentice Hall, 1999.
    .. [2] Daniel W. Griffin, Jae S. Lim "Signal Estimation from
           Modified Short-Time Fourier Transform", IEEE 1984,
           10.1109/TASSP.1984.1164317

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()

    Generate a test signal, a 2 Vrms sine wave whose frequency is slowly
    modulated around 3kHz, corrupted by white noise of exponentially
    decreasing magnitude sampled at 10 kHz.

    >>> fs = 10e3
    >>> N = 1e5
    >>> amp = 2 * np.sqrt(2)
    >>> noise_power = 0.01 * fs / 2
    >>> time = np.arange(N) / float(fs)
    >>> mod = 500*np.cos(2*np.pi*0.25*time)
    >>> carrier = amp * np.sin(2*np.pi*3e3*time + mod)
    >>> noise = rng.normal(scale=np.sqrt(noise_power),
    ...                    size=time.shape)
    >>> noise *= np.exp(-time/5)
    >>> x = carrier + noise

    Compute and plot the STFT's magnitude.

    >>> f, t, Zxx = signal.stft(x, fs, nperseg=1000)
    >>> plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp, shading='gouraud')
    >>> plt.title('STFT Magnitude')
    >>> plt.ylabel('Frequency [Hz]')
    >>> plt.xlabel('Time [sec]')
    >>> plt.show()

    Compare the energy of the signal `x` with the energy of its STFT:

    >>> E_x = sum(x**2) / fs  # Energy of x
    >>> # Calculate a two-sided STFT with PSD scaling:
    >>> f, t, Zxx = signal.stft(x, fs, nperseg=1000, return_onesided=False,
    ...                         scaling='psd')
    >>> # Integrate numerically over abs(Zxx)**2:
    >>> df, dt = f[1] - f[0], t[1] - t[0]
    >>> E_Zxx = sum(np.sum(Zxx.real**2 + Zxx.imag**2, axis=0) * df) * dt
    >>> # The energy is the same, but the numerical errors are quite large:
    >>> np.isclose(E_x, E_Zxx, rtol=1e-2)
    True

    """
    if scaling == 'psd':
        scaling = 'density'
    elif scaling != 'spectrum':
        raise ValueError(f"Parameter {scaling=} not in ['spectrum', 'psd']!")

    freqs, time, Zxx = _spectral_helper(x, x, fs, window, nperseg, noverlap,
                                        nfft, detrend, return_onesided,
                                        scaling=scaling, axis=axis,
                                        mode='stft', boundary=boundary,
                                        padded=padded, nworkers=nworkers)

    return freqs, time, Zxx


def _spectral_helper(x, y, fs=1.0, window='hann', nperseg=None, noverlap=None,
                     nfft=None, detrend='constant', return_onesided=True,
                     scaling='density', axis=-1, mode='psd', boundary=None,
                     padded=False, nworkers=None):
    """Calculate various forms of windowed FFTs for PSD, CSD, etc.

    This is a helper function that implements the commonality between
    the stft, psd, csd, and spectrogram functions. It is not designed to
    be called externally. The windows are not averaged over; the result
    from each window is returned.

    Parameters
    ----------
    x : array_like
        Array or sequence containing the data to be analyzed.
    y : array_like
        Array or sequence containing the data to be analyzed. If this is
        the same object in memory as `x` (i.e. ``_spectral_helper(x,
        x, ...)``), the extra computations are spared.
    fs : float, optional
        Sampling frequency of the time series. Defaults to 1.0.
    window : str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple, it is
        passed to `get_window` to generate the window values, which are
        DFT-even by default. See `get_window` for a list of windows and
        required parameters. If `window` is array_like it will be used
        directly as the window and its length must be nperseg. Defaults
        to a Hann window.
    nperseg : int, optional
        Length of each segment. Defaults to None, but if window is str or
        tuple, is set to 256, and if window is array_like, is set to the
        length of the window.
    noverlap : int, optional
        Number of points to overlap between segments. If `None`,
        ``noverlap = nperseg // 2``. Defaults to `None`.
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired. If
        `None`, the FFT length is `nperseg`. Defaults to `None`.
    detrend : str or function or `False`, optional
        Specifies how to detrend each segment. If `detrend` is a
        string, it is passed as the `type` argument to the `detrend`
        function. If it is a function, it takes a segment and returns a
        detrended segment. If `detrend` is `False`, no detrending is
        done. Defaults to 'constant'.
    return_onesided : bool, optional
        If `True`, return a one-sided spectrum for real data. If
        `False` return a two-sided spectrum. Defaults to `True`, but for
        complex data, a two-sided spectrum is always returned.
    scaling : { 'density', 'spectrum' }, optional
        Selects between computing the cross spectral density ('density')
        where `Pxy` has units of V**2/Hz and computing the cross
        spectrum ('spectrum') where `Pxy` has units of V**2, if `x`
        and `y` are measured in V and `fs` is measured in Hz.
        Defaults to 'density'
    axis : int, optional
        Axis along which the FFTs are computed; the default is over the
        last axis (i.e. ``axis=-1``).
    mode: str {'psd', 'stft'}, optional
        Defines what kind of return values are expected. Defaults to
        'psd'.
    boundary : str or None, optional
        Specifies whether the input signal is extended at both ends, and
        how to generate the new values, in order to center the first
        windowed segment on the first input point. This has the benefit
        of enabling reconstruction of the first input point when the
        employed window function starts at zero. Valid options are
        ``['even', 'odd', 'constant', 'zeros', None]``. Defaults to
        `None`.
    padded : bool, optional
        Specifies whether the input signal is zero-padded at the end to
        make the signal fit exactly into an integer number of window
        segments, so that all of the signal is included in the output.
        Defaults to `False`. Padding occurs after boundary extension, if
        `boundary` is not `None`, and `padded` is `True`.

    Returns
    -------
    freqs : ndarray
        Array of sample frequencies.
    t : ndarray
        Array of times corresponding to each data segment
    result : ndarray
        Array of output data, contents dependent on *mode* kwarg.

    Notes_v01.md
    -----
    Adapted from matplotlib.mlab

    .. versionadded:: 0.16.0
    """
    if mode not in ['psd', 'stft']:
        raise ValueError("Unknown value for mode %s, must be one of: "
                         "{'psd', 'stft'}" % mode)

    boundary_funcs = {'even': even_ext,
                      'odd': odd_ext,
                      'constant': const_ext,
                      'zeros': zero_ext,
                      None: None}

    if boundary not in boundary_funcs:
        raise ValueError("Unknown boundary option '{}', must be one of: {}"
                         .format(boundary, list(boundary_funcs.keys())))

    # If x and y are the same object we can save ourselves some computation.
    same_data = y is x

    if not same_data and mode != 'psd':
        raise ValueError("x and y must be equal if mode is 'stft'")

    axis = int(axis)

    # Ensure we have np.arrays, get outdtype
    x = np.asarray(x)
    if not same_data:
        y = np.asarray(y)
        outdtype = np.result_type(x, y, np.complex64)
    else:
        outdtype = np.result_type(x, np.complex64)

    if not same_data:
        # Check if we can broadcast the outer axes together
        xouter = list(x.shape)
        youter = list(y.shape)
        xouter.pop(axis)
        youter.pop(axis)
        try:
            outershape = np.broadcast(np.empty(xouter), np.empty(youter)).shape
        except ValueError as e:
            raise ValueError('x and y cannot be broadcast together.') from e

    if same_data:
        if x.size == 0:
            return np.empty(x.shape), np.empty(x.shape), np.empty(x.shape)
    else:
        if x.size == 0 or y.size == 0:
            outshape = outershape + (min([x.shape[axis], y.shape[axis]]),)
            emptyout = np.moveaxis(np.empty(outshape), -1, axis)
            return emptyout, emptyout, emptyout

    if x.ndim > 1:
        if axis != -1:
            x = np.moveaxis(x, axis, -1)
            if not same_data and y.ndim > 1:
                y = np.moveaxis(y, axis, -1)

    # Check if x and y are the same length, zero-pad if necessary
    if not same_data:
        if x.shape[-1] != y.shape[-1]:
            if x.shape[-1] < y.shape[-1]:
                pad_shape = list(x.shape)
                pad_shape[-1] = y.shape[-1] - x.shape[-1]
                x = np.concatenate((x, np.zeros(pad_shape)), -1)
            else:
                pad_shape = list(y.shape)
                pad_shape[-1] = x.shape[-1] - y.shape[-1]
                y = np.concatenate((y, np.zeros(pad_shape)), -1)

    if nperseg is not None:  # if specified by user
        nperseg = int(nperseg)
        if nperseg < 1:
            raise ValueError('nperseg must be a positive integer')

    # parse window; if array like, then set nperseg = win.shape
    win, nperseg = _triage_segments(window, nperseg, input_length=x.shape[-1])

    if nfft is None:
        nfft = nperseg
    elif nfft < nperseg:
        raise ValueError('nfft must be greater than or equal to nperseg.')
    else:
        nfft = int(nfft)

    if noverlap is None:
        noverlap = nperseg // 2
    else:
        noverlap = int(noverlap)
    if noverlap >= nperseg:
        raise ValueError('noverlap must be less than nperseg.')
    nstep = nperseg - noverlap

    # Padding occurs after boundary extension, so that the extended signal ends
    # in zeros, instead of introducing an impulse at the end.
    # I.e. if x = [..., 3, 2]
    # extend then pad -> [..., 3, 2, 2, 3, 0, 0, 0]
    # pad then extend -> [..., 3, 2, 0, 0, 0, 2, 3]

    if boundary is not None:
        ext_func = boundary_funcs[boundary]
        x = ext_func(x, nperseg // 2, axis=-1)
        if not same_data:
            y = ext_func(y, nperseg // 2, axis=-1)

    if padded:
        # Pad to integer number of windowed segments
        # I.e make x.shape[-1] = nperseg + (nseg-1)*nstep, with integer nseg
        nadd = (-(x.shape[-1] - nperseg) % nstep) % nperseg
        zeros_shape = list(x.shape[:-1]) + [nadd]
        x = np.concatenate((x, np.zeros(zeros_shape)), axis=-1)
        if not same_data:
            zeros_shape = list(y.shape[:-1]) + [nadd]
            y = np.concatenate((y, np.zeros(zeros_shape)), axis=-1)

    # Handle detrending and window functions
    if not detrend:
        def detrend_func(d):
            return d
    elif not hasattr(detrend, '__call__'):
        def detrend_func(d):
            return _signaltools.detrend(d, type=detrend, axis=-1)
    elif axis != -1:
        # Wrap this function so that it receives a shape that it could
        # reasonably expect to receive.
        def detrend_func(d):
            d = np.moveaxis(d, -1, axis)
            d = detrend(d)
            return np.moveaxis(d, axis, -1)
    else:
        detrend_func = detrend

    if np.result_type(win, np.complex64) != outdtype:
        win = win.astype(outdtype)

    if scaling == 'density':
        scale = 1.0 / (fs * (win * win).sum())
    elif scaling == 'spectrum':
        scale = 1.0 / win.sum() ** 2
    else:
        raise ValueError('Unknown scaling: %r' % scaling)

    if mode == 'stft':
        scale = np.sqrt(scale)

    if return_onesided:
        if np.iscomplexobj(x):
            sides = 'twosided'
            warnings.warn('Input data is complex, switching to '
                          'return_onesided=False')
        else:
            sides = 'onesided'
            if not same_data:
                if np.iscomplexobj(y):
                    sides = 'twosided'
                    warnings.warn('Input data is complex, switching to '
                                  'return_onesided=False')
    else:
        sides = 'twosided'

    if sides == 'twosided':
        freqs = sp_fft.fftfreq(nfft, 1 / fs)
    elif sides == 'onesided':
        freqs = sp_fft.rfftfreq(nfft, 1 / fs)

    # Perform the windowed FFTs
    result = _fft_helper(x, win, detrend_func, nperseg, noverlap, nfft, sides, nworkers=nworkers)

    if not same_data:
        # All the same operations on the y data
        result_y = _fft_helper(y, win, detrend_func, nperseg, noverlap, nfft,
                               sides, nworkers=nworkers)
        result = np.conjugate(result) * result_y
    elif mode == 'psd':
        result = np.conjugate(result) * result

    result *= scale
    if sides == 'onesided' and mode == 'psd':
        if nfft % 2:
            result[..., 1:] *= 2
        else:
            # Last point is unpaired Nyquist freq point, don't double
            result[..., 1:-1] *= 2

    time = np.arange(nperseg / 2, x.shape[-1] - nperseg / 2 + 1,
                     nperseg - noverlap) / float(fs)
    if boundary is not None:
        time -= (nperseg / 2) / fs

    result = result.astype(outdtype)

    # All imaginary parts are zero anyways
    if same_data and mode != 'stft':
        result = result.real

    # Output is going to have new last axis for time/window index, so a
    # negative axis index shifts down one
    if axis < 0:
        axis -= 1

    # Roll frequency axis back to axis where the data came from
    result = np.moveaxis(result, -1, axis)

    return freqs, time, result


def _fft_helper(x, win, detrend_func, nperseg, noverlap, nfft, sides, nworkers=None):
    """
    Calculate windowed FFT, for internal use by
    `scipy.signal._spectral_helper`.

    This is a helper function that does the main FFT calculation for
    `_spectral helper`. All input validation is performed there, and the
    data axis is assumed to be the last axis of x. It is not designed to
    be called externally. The windows are not averaged over; the result
    from each window is returned.

    Returns
    -------
    result : ndarray
        Array of FFT data

    Notes_v01.md
    -----
    Adapted from matplotlib.mlab

    .. versionadded:: 0.16.0
    """
    # Created strided array of data segments
    if nperseg == 1 and noverlap == 0:
        result = x[..., np.newaxis]
    else:
        # https://stackoverflow.com/a/5568169
        step = nperseg - noverlap
        shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // step, nperseg)
        strides = x.strides[:-1] + (step * x.strides[-1], x.strides[-1])
        result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                                 strides=strides)

    # Detrend each data segment individually
    result = detrend_func(result)

    # Apply window by multiplication
    result = win * result

    # Perform the fft. Acts on last axis by default. Zero-pads automatically
    if sides == 'twosided':
        # func = sp_fft.fft()
        result = sp_fft.fft(result, n=nfft, workers=nworkers)
    else:
        result = result.real
        # func = sp_fft.rfft()
        result = sp_fft.rfft(result, n=nfft, workers=nworkers)

    return result
