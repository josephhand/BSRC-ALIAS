# Scratchwork

This folder serves as a place for intermediate experimentation of algorithms.
Some of these will eventually end up in the main module.

Here is a quick summary of what all the notebooks in this folder contain:

- `continuum_normalize.ipynb`: A notebook implementing the continuum
  normalization algorithm used by ALIAS. This has already been implemented in
  the `alias.continuum_normalization` submodule.
- `injection_test.ipynb`: A first attempt at injection testing. Formerly
- implemented in the `alias_injection` submodule.
- `injection_test_2.ipynb`: The second attempt at injection testing.
- `line_detection.ipynb`: The first attempt at detecting lines and excluding
  lines common between many spectra.
- `plotting_spectra.ipynb`: Experiments with plotting spectra to understand
  their structure.
- `statistical_test.ipynb`: An accessment of the viability of using statistical
  properties such as median absolute deviation, and percentile ranges at
  detecting regions of atmospheric interference
- `target_selection.ipynb`: Demontration of the target selection used to obtain
  the initial dataset.