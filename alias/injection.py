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

import alias
import numpy as np
import pandas as pd

pd.set_option("mode.chained_assignment", None)

def _gaussian(x, amp, mean, fwhm):
    return amp * 2.71828**(-2.773 * ((x - mean)/fwhm)**2)

def _inject_line(spectrum, wavelength, amplitude, width):
    wave = spectrum['WAVE']
    flux = spectrum['FLUX']

    return flux + np.nanmedian(flux) * _gaussian(wave, amplitude, wavelength, width)

def inject_dataframe(data, candidates, spec_id, wavelength, amplitude, width):

    # Make copies of data
    data_new = data.copy(deep=True)
    
    # Note that candidates from the spectrum being modified are excluded
    candidates_new = candidates[~(candidates['SPECTRUM_ID'] == spec_id)].copy()

    # Inject the line
    new_flux = _inject_line(data.loc[spec_id], wavelength, amplitude, width)

    # Update aother data products
    new_norm_flux, new_norm_flux_error = alias.continuum_normalize_one(new_flux, data['FLUX_ERR'][spec_id])

    row = data_new.loc[spec_id]
    row['FLUX'] = list(new_flux)
    row['RESIDUALS'] = data_new['RESIDUALS'][spec_id] - data_new['NORM_FLUX'][spec_id] + new_norm_flux
    row['NORM_FLUX'] = new_norm_flux
    data_new.loc[spec_id] = row
    
    # Update candidates list
    candidates_modified = alias.detect_one(data_new, spec_id, 0.05)
    alias.characterize_all(data_new, candidates_modified)

    return data_new, candidates_modified