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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.io import fits
import scipy.signal

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


class Dataset:
    '''A class to represent a collection of APOGEE spectra.

    This class is designed to streamline the process of loading and collating
    APOGEE spectra.

    Once initialized, a Dataset object has the following members that can be accessed.

    - **targets** A 2D numpy array containing properties of stars, extracted from the fits file

    - **wave** A 1D numpy array containing the wavelengths of each spectral element in every
      spectrum, measured in angstroms.
    
    - **flux** A 2D numpy array with the relative flux values of each spectrum. These values
      are already normalized to have a median of 1.
    the
    - **ivar** A 2D numpy array containing the inverse-variances of each spectral element.
    '''

    def __init__(self, wave, flux, ivar):
        self.wave = wave
        self.flux = flux
        self.ivar = ivar


def loadDataset(urls):
    '''Load a dataset from a list of urls or filepaths.'''
    wave = None
    flux = []
    ivar = []

    for i in range(len(urls)):
        url = urls[i]
        hdul = fits.open(url)

        spec_flux_parts = np.array(hdul[1].data)
        spec_ivar_parts = np.array(hdul[2].data)**-2

        mask = np.isnan(spec_flux_parts) | np.isnan(spec_ivar_parts) | np.isinf(spec_ivar_parts)
        masked_flux = np.ma.MaskedArray(spec_flux_parts, mask=mask)
        masked_ivar = np.ma.MaskedArray(spec_ivar_parts, mask=mask)

        spec_flux = np.average(
            masked_flux,
            axis=0, 
            weights=masked_ivar
        )
        spec_ivar = np.sum(masked_ivar, axis=0)

        flux.append(spec_flux)
        ivar.append(spec_ivar)

        if i == 0:
            wave = 10**(hdul[1].header['CRVAL1'] + (hdul[1].header['CDELT1']
                * np.arange(hdul[1].data.shape[1])))

    return Dataset(wave, np.array(flux), np.array(ivar))


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
def continuum_normalize(ds):
    '''Continuum normalize an array of spectra.'''
    continuums = np.array([ _extract_continuum(f) for f in ds.flux ])[:,range(len(ds.flux[0]))]
    return Dataset(ds.wave, ds.flux/continuums, ds.ivar*continuums**2)

def get_residuals(ds):
    masked_flux = np.ma.filled(np.ma.MaskedArray(ds.flux, ds.flux < 0.05), np.nan)
    median_flux = np.nanmedian(masked_flux, axis=0)
    return Dataset(ds.wave, ds.flux - median_flux, ds.ivar)

def detect(wave, flux, ivar):
    peaks = scipy.signal.find_peaks(flux, height = 0.05)[0]
    return peaks


def detect_all(ds):

    all_detections = []
    
    for n in range(len(ds.flux)):
        peaks = detect(ds.wave, ds.flux[n], ds.ivar[n])
        for peak in peaks:
            all_detections.append((n,peak))

    return np.array(all_detections, dtype=int)


def _chi2_lsf(y, y_err, lsfx, lsfy, amp, center_pix):
    lsf = np.interp(range(len(y)), lsfx + center_pix, amp*lsfy)
    return np.sum(((y - lsf)/y_err)**2)


def _characterize_single(wave, flux, ivar, amp):
    center_idx = np.linspace(len(wave)/2 - 1, len(wave)/2, 64)
    
    chi2 = [ _chi2_lsf(flux, ivar**-0.5, default_lsf.x, np.array(default_lsf.y), 0.3, center) for center in center_idx ]
    
    best_idx = center_idx[np.argmin(chi2)]
    best_wl = np.interp(best_idx, range(len(wave)), wave)

    # We can use our approximate guess of the amplitude to get a range to look for an improved amplitude
    amps = np.linspace(amp * 0.7, amp*1.4, 64)
    
    chi2 = [ 
        _chi2_lsf(flux, ivar**-0.5, default_lsf.x, np.array(default_lsf.y), amp, best_idx)
        for amp in amps
    ]
    
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


def characterize(wave, flux, ivar, peak):
    peak_w = wave[peak-10:peak+11]
    peak_f = flux[peak-10:peak+11]
    peak_i = ivar[peak-10:peak+11]
    nan_filter = np.isnan(peak_f) | np.isnan(peak_w) | np.isnan(peak_i)
    return _characterize_single(peak_w[~nan_filter], peak_f[~nan_filter], peak_i[~nan_filter], flux[peak])

def characterize_all(ds, detections):
    characterizations = np.array([
        list(d) + list(characterize(ds.wave, ds.flux[int(d[0])], ds.ivar[int(d[0])], int(d[1])))
        for d in detections
    ])
    return characterizations

#def auto_classify(ds, detections, characterizations):
    
