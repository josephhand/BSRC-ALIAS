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

import numpy as np
from astropy.io import fits

class Dataset:
    '''A class to represent a collection of APOGEE spectra.

    This class is designed to streamline the process of loading and combining
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
        #spec_flux = spec_flux_parts[0]
        #spec_ivar = spec_ivar_parts[0]

        flux.append(spec_flux)
        ivar.append(spec_ivar)

        if i == 0:
            wave = 10**(hdul[1].header['CRVAL1'] + (hdul[1].header['CDELT1']
                * np.arange(hdul[1].data.shape[1])))

    return Dataset(wave, np.array(flux), np.array(ivar))
            
        






