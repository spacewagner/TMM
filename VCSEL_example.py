# %%

"""
TODO

mark effective cavity length in plot
print report to txt or table
Use TMM to compare devices e.g. mode confinement over cavity length, DBR Pairs....
expand lifetime analysis to predict dynamic properties
range and resolution for cavity dip analysis has to be auto-adjusted
use separat mesh for cavity analysis to save time

define analysis methods as classes. More complex methods should inherit attributes from the simpler ones. classes should include plot and print methods

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from TMM.structure_builder import (
    plot_structure,
    apply_AR_coating,
    build_VCSEL_structure,
    wavelength_arr_adaptive_mesh,
    get_VCSEL_structure,
    build_DBR_structure,
    apply_etch,
    interpolate_structure,
    build_active_region,
)
from TMM.field_solver import (
    calculate_optical_properties,
    calculate_electrical_field,
)

from TMM.analysis import (
    analyse_VCSEL,
    analyse_AR_coating,
    analyse_etching,
    calculate_temperature_shift,
    analyse_cavity_dip,
    analyse_reflectivity_tuning,
    analyse_VCSELs_DBRs,
    analyse_electrical_field,
    analyse_VCSEL_lifetime_tuning,
    VCSEL_embedding_active_region,
)

from TMM.optimizations import optimize_embedding_thickness

from TMM.optics_utils import refractive_index_AlGaAs_at_1310, refractive_index_SiO2

# %%
"""
Build and plot a VCSEL Structure.

"""
target_wavelength = 1310e-9
GaAs = refractive_index_AlGaAs_at_1310(0)
AlAs = refractive_index_AlGaAs_at_1310(100)

n_bottom_1 = GaAs
n_bottom_2 = AlAs
N_bottom = 35.5
n_top_1 = GaAs
n_top_2 = AlAs
N_top = 21.5
n_cavity = 3.205
n_substrate = GaAs
n_air = 1

VCSEL = build_VCSEL_structure(
    n_bottom_1,
    n_bottom_2,
    N_bottom,
    n_top_1,
    n_top_2,
    N_top,
    n_cavity,
    n_substrate,
    n_air,
    target_wavelength,
    N_cavity=5,
)

plot_structure(VCSEL)

# %% Full Analysis

analyse_VCSEL_results = analyse_VCSEL(VCSEL, target_wavelength, fine_range=1e-9)

# %% Analyse Spectra

"""
Analyse VCSELs spectral response.

Set a wavelength range for the spectra by using the adaptive mesh function. This allows to have higher resolution close to the target wavelength.

"""

# set adaptive wavelength mesh
wavelength_arr = wavelength_arr_adaptive_mesh(
    target_wavelength - 100e-9,
    target_wavelength + 100e-9,
    target_wavelength,
    fine_range=0.2e-9,
    mesh_size=0.001e-9,
)

# run analysis
VCSEL_optical_properties_result = calculate_optical_properties(VCSEL, wavelength_arr)

# %%
"""
analyse the electrical field distribution within the VCSEL.

"""

VCSEL_field_properties_results = analyse_electrical_field(VCSEL, target_wavelength)

# %% Advanced functions

"""
Recover the structures parameters.
"""

(
    n_bottom_1,
    n_bottom_2,
    N_bottom,
    n_top_1,
    n_top_2,
    N_top,
    n_cavity,
    n_substrate,
    n_air,
) = get_VCSEL_structure(VCSEL)


# %%

"""
Build DBR structures.

"""

DBR_bottom = build_DBR_structure(
    n_bottom_1, n_bottom_2, N_bottom, target_wavelength, n_substrate, n_cavity
)
plot_structure(DBR_bottom)
plt.show()

DBR_top = build_DBR_structure(
    n_top_1, n_top_2, N_top, target_wavelength, n_cavity, n_air
)
plot_structure(DBR_top)
plt.show()


# %% analyse DBRs

"""
analyse DBRs directliy from a defined VCSEL structure.

"""

VCSELs_DBRs_properties = analyse_VCSELs_DBRs(VCSEL, target_wavelength, wavelength_arr)
# %% Resonator Quality

"""
Analyse the cavity response and the resonators quality.

"""

VCSEL_cavity_properties_result = analyse_cavity_dip(
    wavelength_arr,
    VCSEL_optical_properties_result.R_arr,
    VCSEL_optical_properties_result.T_arr,
    target_wavelength,
    fine_range=0.1e-9,
)

# %% Reflectivity tuning

"""
Investigate reflectivity tuning on the VCSELs structure by adding an anti reflective coating on the top DBR or by etching it partially. Default material for AR Coating is SiO2 with n = 1.45.

"""

VCSEL_coating_properties, VCSEL_etching_properties = analyse_reflectivity_tuning(
    VCSEL, target_wavelength
)

# %% Lifetime tuning in detail

"""
Apply etch to the investigated structure.

"""
VCSEL_etch = apply_etch(VCSEL, 100e-9)
plot_structure(VCSEL_etch)
plt.show()

"""
Investigate the effect of etching on reflectivity.

"""

VCSEL_etching_properties = analyse_etching(VCSEL, target_wavelength)

"""
Apply an AR coating.

"""
n_coating = refractive_index_SiO2(target_wavelength)
VCSEL_AR = apply_AR_coating(VCSEL, n_coating, 100e-9)
plot_structure(VCSEL_AR)
plt.show()

"""
Investigate the effect of AR coating on reflectivity.


"""
VCSEL_coating_properties = analyse_AR_coating(VCSEL, target_wavelength)

# %% Temperature tuning

"""
analyse the shift of the cavity wavelength with temperature. Default temperature coefficient is 0.061 nm/K.

"""

T_arr = np.linspace(300, 400, 5)

VCSEL_temperature_properties = calculate_temperature_shift(
    VCSEL, target_wavelength, T_arr
)

# %% Lifetime Tuning
n_coating = refractive_index_SiO2(target_wavelength)
alpha_i_arr = [0, 5e2, 10e2, 20e2]
VCSEL_cavity_tuning_properties = analyse_VCSEL_lifetime_tuning(
    VCSEL,
    target_wavelength,
    n_coating=n_coating,
    alpha_i_arr=alpha_i_arr,
    resolution=2e-9,
)

# %% Define active region
n_quantum_wells = [3.53, 3.49] * 24
d_quantum_wells = [0.8e-9, 0.2e-9] * 24

n_arr = [3.53, 3.49, 3.24, 3.49, *n_quantum_wells, 3.49, 3.205, 3.53, 3.205]
d_arr = [30e-9, 15e-9, 35e-9, 4e-9, *d_quantum_wells, 4e-9, 50e-9, 10e-9, 21e-9]

active_region = build_active_region(n_arr, d_arr)
plot_structure(active_region)
# %% Find cavity length for optimum confinement
results = optimize_embedding_thickness(
    VCSEL,
    active_region,
    target_wavelength=target_wavelength,
    d_min=0e-9,
    d_max=500e-9,
)

# %% Build VCSEL with active region, with optimum cavity length
VCSEL_modified = VCSEL_embedding_active_region(
    VCSEL, active_region, results.d_optimum_arr[0]
)

results_electrical_field = calculate_electrical_field(
    VCSEL_modified, target_wavelength, Plot=True
)
