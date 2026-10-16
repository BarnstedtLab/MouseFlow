#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pyTreadMouse functions to analyse camera data

Created on Wed May  8 14:31:51 2019
@author: Oliver Barnstedt
"""


import os.path
import pandas as pd
from tqdm import tqdm
import cv2
from matplotlib import pyplot as plt
import numpy as np
from scipy import optimize, signal

plt.interactive(False)


# preallocate empty array and assign slice by chrisaycock
def shift5(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

def dlc_pointdistance2(point1, point2):
    return pd.Series(np.linalg.norm((point2 - point1), axis=1))


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:
    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

#    if x.ndim != 1:
#        raise ValueError, "smooth only accepts 1 dimension arrays."
#
#    if x.size < window_len:
#        raise ValueError, "Input vector needs to be bigger than window size."
    if window_len<3:
        return x

#    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
#        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.'+window+'(window_len)')
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y


def hilbert_peaks(x, fps, fc=None, butterN=5, peakprom=1.5):
    env = np.abs(signal.hilbert(x))
    nyquist = fps/2.0
    # Low-Pass Butter filter:
    # If no cutoff given or too high, use half‑Nyquist
    if fc is None or fc >= nyquist:
        fc_used = nyquist * 0.5
    else:
        fc_used = fc
    w = fc_used / nyquist  # always < 1
    b, a = signal.butter(butterN, w, 'low')
    shift_amt = int(0.6 * fps)
    mouth_envfilt = pd.Series(signal.filtfilt(b, a, env)).shift(shift_amt)
    peaks = signal.find_peaks(mouth_envfilt, prominence=peakprom)[0]
    peakbool = np.zeros(len(x))
    peakbool[peaks] = 1
    return mouth_envfilt, peakbool

def freq_analysis2(x, fps, rollwin=75, min_periods=50, conf=0.5):
    if isinstance(x, pd.DataFrame) and {'x', 'likelihood'}.issubset(x.columns):
        x_filtered = x.copy()
        x_filtered.loc[x_filtered['likelihood'] < conf, 'x'] = np.nan
        y = x_filtered['x']
    elif isinstance(x, pd.DataFrame) and 'x' in x.columns:
        y = x['x']
    elif isinstance(x, pd.Series):
        y = x
    else:
        raise ValueError("Input must be a pandas Series or a DataFrame containing an 'x' column (and optionally 'likelihood').")
    xinterp = y.interpolate(method='linear')
    xz = (xinterp - xinterp.min()) / (xinterp - xinterp.min()).max()
    print('Fitting sine...')
    w = xz.rolling(int(rollwin), center=True, min_periods=min_periods).apply(fit_sin_w)
    error = xz.rolling(int(rollwin), center=True, min_periods=min_periods).apply(fit_sin_error)
    w_clean = w.copy()
    w_clean[w_clean.diff().abs()>(.75/fps)] = np.nan
    w_clean[error>(fps/100)] = np.nan
    w_clean[w_clean<0] = np.nan
    w_clean = w_clean.interpolate(method='polynomial', order=3, limit=int(fps*2))
    w_smooth = (w_clean.rolling(window=int(rollwin/3), win_type='gaussian', center=True, min_periods=int(min_periods/3),).mean(std=rollwin/3))
    f_smooth = w_smooth/(2.*np.pi)*fps
    return f_smooth


def fit_sin_w(yy):  # omega
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(range(len(yy)))
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1] - tt[0]))  # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:]) + 1])  # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2. ** 0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2. * np.pi * guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):
        return A * np.sin(w * t + p) + c
    try:
        popt, pcov = optimize.curve_fit(sinfunc, tt, yy, p0=guess)
        A, w, p, c = popt
        return w
    except:
        return np.nan



def fit_sin_error(yy):  # mean SD error
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(range(len(yy)))
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1] - tt[0]))  # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:]) + 1])  # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2. ** 0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2. * np.pi * guess_freq, 0., guess_offset])
    def sinfunc(t, A, w, p, c):
        return A * np.sin(w * t + p) + c
    try:
        popt, pcov = optimize.curve_fit(sinfunc, tt, yy, p0=guess)
        A, w, p, c = popt
        return np.mean(np.sqrt(np.diag(pcov)))
    except:
        return np.nan
