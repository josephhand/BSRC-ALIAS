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

'''Small library for continuum normalizing APOGEE spectra'''

import numpy as np

import tqdm.autonotebook as tqdm

import math

def _continuum(flux, segment_len=100):
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
def continuum_normalize(flux, ivar):
    '''Continuum normalize an array of spectra.'''
    continuums = np.array([ _continuum(f) for f in flux ])[:,range(len(flux[0]))]
    return flux/continuums, ivar*continuums**2, continuums