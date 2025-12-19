# %%

"""
TODO

mark effective cavity length in plot
print report to txt or table
return results in better format


"""

import numpy as np
import matplotlib.pyplot as plt

from TMM.structure_builder import (
    plot_structure,
    apply_AR_coating,
    build_VCSEL_structure,
    wavelength_arr_adaptive_mesh,
    get_VCSEL_structure,
    build_DBR_structure,
    apply_etch,
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
)

# %%
"""
Build and plot a VCSEL Structure.

"""

GaAs = 3.52
AlAs = 2.95

n_bottom_1 = AlAs
n_bottom_2 = GaAs
N_bottom = 20.5
n_top_1 = AlAs
n_top_2 = GaAs
N_top = 20.0
n_cavity = GaAs
n_substrate = GaAs
n_air = 1
target_wavelength = 940e-9

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
    N_cavity=2,
)

plot_structure(VCSEL)

# %% Full Analysis

analyse_VCSEL(VCSEL, target_wavelength)

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
    fine_range=5e-9,
    mesh_size=0.01e-9,
)

# run analysis
wavelength_arr, r_arr, t_arr, phase_arr = calculate_optical_properties(
    VCSEL, wavelength_arr
)

# %%
"""
analyse the electrical field distribution within the VCSEL.

"""

field_positions, field_values, n_field_arr, Gamma_z, alpha_i = analyse_electrical_field(
    VCSEL, target_wavelength
)

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

(
    DBR_bottom_r_arr,
    DBR_bottom_t_arr,
    DBR_bottom_phase_arr,
    DBR_bottom_r_at_target,
    DBR_bottom_phase_at_target,
    DBR_bottom_stopband_width,
    DBR_top_r_arr,
    DBR_top_t_arr,
    DBR_top_phase_arr,
    DBR_top_r_at_target,
    DBR_top_phase_at_target,
    DBR_top_stopband_width,
    L_eff,
    alpha_m,
    cavity_lifetime,
) = analyse_VCSELs_DBRs(VCSEL, target_wavelength, wavelength_arr)
# %% Resonator Quality

"""
Analyse the cavity response and the resonators quality.

"""

(
    wavelength_arr_ROI,
    t_arr_ROI_leveled,
    cavity_resonance_wavelength,
    cavity_resonance_reflectivity,
    cavity_gain,
    FWHM,
    Q_factor,
    cavity_lifetime,
    halfmax,
    hw_start,
    hw_stop,
) = analyse_cavity_dip(wavelength_arr, r_arr, t_arr, target_wavelength)

# %% Reflectivity tuning

"""
Investigate reflectivity tuning on the VCSELs structure by adding an anti reflective coating on the top DBR or by etching it partially. Default material for AR Coating is SiO2 with n = 1.45.

"""

(
    d_coating_arr,
    R_coating_arr,
    R_coating_tuning_range,
    n_AR_coating,
    d_etch_arr,
    R_etch_arr,
    R_etch_tuning_range,
) = analyse_reflectivity_tuning(VCSEL, target_wavelength)

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

d_etch_arr, R_etch_arr, R_etch_tuning_range = analyse_etching(VCSEL, target_wavelength)

"""
Apply an AR coating.

"""

VCSEL_AR = apply_AR_coating(VCSEL, 1.45, 100e-9)
plot_structure(VCSEL_AR)
plt.show()

"""
Investigate the effect of AR coating on reflectivity.

"""

d_coating_arr, R_coating_arr, R_coating_tuning_range = analyse_AR_coating(
    VCSEL, target_wavelength
)

# %% Temperature tuning

"""
analyse the shift of the cavity wavelength with temperature. Default temperature coefficient is 0.061 nm/K.

"""

T_arr = np.linspace(300, 400, 5)

(
    T_arr,
    wavelength_T_theory_arr,
    wavelength_arr_arr,
    cavity_resonance_arr,
    temperature_coefficent,
) = calculate_temperature_shift(VCSEL, target_wavelength, T_arr)

# %% Lifetime Tuning

(
    d_coating_arr,
    R_coating_arr,
    R_coating_tuning_range,
    n_coating,
    L_eff_coating_arr,
    n_cavity_eff_coating_arr,
    v_gr_coating_arr,
    alpha_m_coating_arr,
    alpha_m_coating_range,
    photon_lifetime_coating_arr,
    photon_lifetime_coating_range,
    d_etch_arr,
    R_etch_arr,
    R_etch_tuning_range,
    L_eff_etch_arr,
    n_cavity_eff_etch_arr,
    v_gr_etch_arr,
    alpha_m_etch_arr,
    alpha_m_etch_range,
    photon_lifetime_etch_arr,
    photon_lifetime_etch_range,
) = analyse_VCSEL_lifetime_tuning(
    VCSEL, target_wavelength, n_coating=1.45, alpha_i=5e2, resolution=5e-9
)

# %%
