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

'''Tools for injecting LASER technosignatures into APOGEE spectra.'''

import alias

import numpy as np
from astropy.io import fits

class LSF:
    '''Class to represent a line spread function (LSF).'''
    def __init__(self, lsfx, lsfy):
        self.x = lsfx
        self.y = lsfy

'''The default LSF derived from APOGEE DR12.'''
defaultLSF = LSF(np.linspace(-7.,7.,43), 
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

#apogee_lsfx, apogee_lsfy = _loadLSF()

def _createLaserSignature(wave, lsf, wl):
    idx = np.interp(wl, wave, range(len(wave)))
    line = np.interp(np.array(range(len(wave)))-idx, lsf.x, lsf.y)
    return line

def inject(dataset, lsf, specId, wl, amp):
    nflux = np.copy(dataset.flux)
    nflux[specId] += _createLaserSignature(dataset.wave, lsf, wl)*amp
    return alias.Dataset(dataset.wave, nflux, dataset.ivar)

def inject_random(dataset, lsf, amp, num=1):
    if not amp is tuple:
        amp = (amp, amp)
    nflux = dataset.flux
    pos = []
    for i in range(num):
        specId = np.rand.randrange(len(dataset.flux))
        line_amp = np.uniform(*amp)
        line_wl = np.uniform(dataset.wave[0], dataset.wave[-1])
        nflux[specId] += _createLaserSignature(dataset.wave, lsf, line_wl)*line_amp
        pos += (specId, line_wl, line_amp)
    return alias.Dataset(dataset.wave, nflux, dataset.ivar)
        

