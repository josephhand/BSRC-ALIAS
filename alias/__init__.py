# (C) 2023 Joseph Hand.
#
# This file is part of ALIAS.
#
# ALIAS is free software: you can redistribute it and/or modify it under the terms of
# the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
#
# ALIAS is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along with
# ALIAS. If not, see <https://www.gnu.org/licenses/>. 

'''ALIAS: Anomalous Lines In APOGEE Spectra'''

__version__  = '0.0.2'

import math
import random as rand

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.io import fits
import scipy.signal
import scipy.optimize

import tqdm.autonotebook as tqdm


class LSF:
    '''Class to represent a line spread function (LSF).'''
    def __init__(self, lsfx, lsfy):
        self.x = lsfx
        self.y = lsfy


'''The default LSF derived from APOGEE DR12 data.

This variable is provided for users not in possession of a more recent LSF.
Users with a more recent one should use that instead.'''
default_lsf = LSF(np.linspace(-7.,7.,43), 
                 [
                     0.00308409, 0.00349727, 0.00405324, 0.00471973,
                     0.00561687, 0.00755368, 0.01002816, 0.01260949,
                     0.01570783, 0.02114526, 0.03197088, 0.05200233,
                     0.08584419, 0.13909110, 0.21720967, 0.32272260,
                     0.45251839, 0.60676233, 0.76493717, 0.89244588,
                     0.97567601, 1.00000000, 0.96041723, 0.86549600,
                     0.73198544, 0.58318216, 0.43714065, 0.30872274,
                     0.21824080, 0.15021121, 0.09954796, 0.06641710,
                     0.04516253, 0.03132488, 0.02248034, 0.01609376,
                     0.01115971, 0.00800559, 0.00596385, 0.00467526,
                     0.00387761, 0.00336630, 0.00304458
                 ]
)

def _extract_continuum(flux, segment_len=100):
    '''Extract the continuum from a spectrum.'''
    flux = np.concatenate((flux, [math.nan]*(segment_len - len(flux) % segment_len)))
    
    f_idx = np.array([ range(segment_len*n, segment_len*n+segment_len) for n in range(0, int(len(flux)/segment_len)) ])
    flux_segments = flux[f_idx]
    
    max_perc = np.nanpercentile(flux_segments, 80, axis=1)
    min_perc = np.nanpercentile(flux_segments, 70, axis=1)
    
    all_continuum_pixels = np.concatenate([ 
        np.where((flux_segments[n] > min_perc[n]) & (flux_segments[n] < max_perc[n]))[0] + segment_len*n 
        for n in range(len(flux_segments))
    ])

    cont1 = all_continuum_pixels[(all_continuum_pixels < 3400)]
    fit1 = np.polyfit(cont1, flux[cont1], 6)
    cont2 = all_continuum_pixels[(all_continuum_pixels > 3400) & (all_continuum_pixels < 6250)]
    fit2 = np.polyfit(cont2, flux[cont2], 6)
    cont3 = all_continuum_pixels[(all_continuum_pixels > 6250)]
    fit3 = np.polyfit(cont3, flux[cont3], 6)
        
    cont1 = np.polyval(fit1, range(len(flux)))
    cont2 = np.polyval(fit2, range(len(flux)))
    cont3 = np.polyval(fit3, range(len(flux)))
    
    continuum = np.concatenate((cont1[:3400], cont2[3400:6250], cont3[6250:]))

    return continuum


# Function to continuum normalize a list of spectra
def continuum_normalize(data, flux_header='FLUX', flux_error_header='FLUX_ERR', norm_flux_header='NORM_FLUX', norm_flux_err_header='NORM_FLUX_ERR'):
    '''Continuum normalize an array of spectra.'''
    flux = data[flux_header]
    flux_error = data[flux_error_header]
    continuums = np.array([ _extract_continuum(f) for f in flux ])[:,range(len(flux[0]))]
    norm_flux = [ flux[i]/continuums[i,:] for i in range(len(flux)) ]
    norm_flux_error = [ flux_error[i]/continuums[i,:] for i in range(len(flux)) ]
    data[norm_flux_header] = norm_flux
    data[norm_flux_err_header] = norm_flux_error

def get_residuals(data, norm_flux_header='NORM_FLUX', residual_header='RESIDUALS'):
    flux = np.array(list(data[norm_flux_header]), dtype=float)
    masked_flux = np.ma.filled(np.ma.MaskedArray(flux, flux < 0.05), np.nan)
    median_flux = np.nanmedian(masked_flux, axis=0)
    data[residual_header] = list(flux - median_flux)


def detect(data, threshold, residual_header='RESIDUALS'):

    all_detections = []
    
    residuals = np.array(list(data[residual_header]), dtype=float)
    
    for n, residual in enumerate(residuals):
        peaks = scipy.signal.find_peaks(residual, height = threshold)[0]
        for peak in peaks:
            all_detections.append((n,peak))

    dataframe = pd.DataFrame(all_detections, columns=('SPECTRUM_ID', 'PIXEL'))

    return dataframe


def gaussian_fit(wave, flux, peak_pixel):

    # Find lower bound of event
    l_bound = peak_pixel
    while flux[l_bound] > flux[l_bound - 1]:
        l_bound = l_bound - 1
        
    # Find upper bound of event
    u_bound = peak_pixel + 1
    while flux[u_bound - 1] > flux[u_bound]:
        u_bound = u_bound + 1
    
    print((l_bound, u_bound))

    wave_event = wave[l_bound:u_bound] - wave[peak_pixel]
    flux_event = flux[l_bound:u_bound]

    def gaussian(x, amp, mean, sigma):
        return amp * 2.71828**(-(x-mean)**2/(2*sigma**2))

    params = scipy.optimize.curve_fit(
        gaussian,
        wave_event,
        flux_event,
        bounds = (
            (0, wave[peak_pixel-2]-wave[peak_pixel], 0),
            (np.inf, wave[peak_pixel+2]-wave[peak_pixel], np.inf)
        )
    )[0]

    mse = np.mean((flux_event - gaussian(wave_event, *params)) ** 2)

    return list(params).append(mse)


def characterize_all(ds, detections):
    characterizations = np.array([
        list(d) + list(characterize(ds.wave, ds.flux[int(d[0])], ds.ivar[int(d[0])], int(d[1])))
        for d in detections
    ])
    return characterizations

#def auto_classify(ds, detections, characterizations):
    
