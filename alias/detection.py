import alias
import alias.injection as inj
import alias.continuum_normalization as cn

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.signal

import numpy as np
from astropy.io import fits

import random as rand

import tqdm.autonotebook as tqdm

def _chi2_lsf(y, y_err, lsfx, lsfy, amp, center_pix):
    lsf = np.interp(range(len(y)), lsfx + center_pix, amp*lsfy)
    return np.sum(((y - lsf)/y_err)**2)

def _characterize_single(wave, flux, ivar, amp):
    center_idx = np.linspace(len(wave)/2 - 1, len(wave)/2, 64)
    
    chi2 = [ _chi2_lsf(flux, ivar**-0.5, inj.default_lsf.x, np.array(inj.default_lsf.y), 0.3, center) for center in center_idx ]
    
    best_idx = center_idx[np.argmin(chi2)]
    best_wl = np.interp(best_idx, range(len(wave)), wave)

    # We can use our approximate guess of the amplitude to get a range to look for an improved amplitude
    amps = np.linspace(amp * 0.7, amp*1.4, 64)
    
    chi2 = [ _chi2_lsf(flux, ivar**-0.5, inj.default_lsf.x, np.array(inj.default_lsf.y), amp, best_idx) for amp in amps ]
    
    best_amplitude = amps[np.argmin(chi2)]

    wave_idx = int(len(wave)/2)

    min = wave_idx-1
    while flux[min] > best_amplitude/2 and min > 0:
        min = min-1
    if min == 0:
        wl_l = wave[0]
    else:
        wl_l = np.interp(best_amplitude/2, flux[min:min+2], wave[min:min+2])
    
    max = wave_idx+1
    while flux[max] > best_amplitude/2 and max < len(wave) - 1:
        max = max+1
    if max == 0:
        wl_h = wave[-1]
    else:
        wl_h = np.interp(best_amplitude/2, np.flip(flux[max-1:max+1]), np.flip(wave[max-1:max+1]))

    return best_wl, best_amplitude, wl_h-wl_l

def detect_all(wave, flux, ivar):

    all_detections = []
    
    for n in range(len(flux)):
        peaks = detect(wave, flux[n], ivar[n])
        for peak in peaks:
            all_detections.append((n,peak))

    return np.array(all_detections, dtype=int)

def detect(wave, flux, ivar):
    peaks = scipy.signal.find_peaks(flux, height = 0.05)[0]
    return peaks

def characterize(wave, flux, ivar, peak):
    peak_w = wave[peak-10:peak+11]
    peak_f = flux[peak-10:peak+11]
    peak_i = ivar[peak-10:peak+11]
    nan_filter = np.isnan(peak_f) | np.isnan(peak_w) | np.isnan(peak_i)
    return _characterize_single(peak_w[~nan_filter], peak_f[~nan_filter], peak_i[~nan_filter], flux[peak])

