# %%

"""
TODO

Compare to examples from Chuang and Coldren


"""

import matplotlib.pyplot as plt

from TMM.structure_builder import (
    build_DBR_structure,
    plot_structure,
    wavelength_arr_adaptive_mesh,
    flip_structure,
)
from TMM.field_solver import calculate_optical_properties
from TMM.optics_utils import R_theoretical

from TMM.analysis import analyze_electrical_field, analyze_DBR

# %%

"""

Define DBR structure

"""

GaAs = 3.52
AlAs = 2.95

target_wavelength = 980e-9

# Set wavelength range for the spectra by using the adaptive mesh function
wavelength_arr = wavelength_arr_adaptive_mesh(
    target_wavelength - 100e-9, target_wavelength + 100e-9, target_wavelength, 5e-9
)

# Build DBR structure
DBR = build_DBR_structure(
    GaAs, AlAs, 20.5, target_wavelength, n_substrate=GaAs, n_air=1
)

# Plot the DBR structure
plot_structure(DBR)

# run full analysis on the DBR
(
    DBR_r_arr,
    DBR_t_arr,
    DBR_phase_arr,
    DBR_r_at_target,
    DBR_phase_at_target,
    DBR_stopband_width,
    DBR_field_positions,
    DBR_field_values,
    DBR_n_field_arr,
    l_eff,
    interface_position,
) = analyze_DBR(DBR, target_wavelength, wavelength_arr)

# %%

"""
Flip the structure to imitate incident wave from the opposite side.

"""

DBR_flipped = flip_structure(DBR)

(
    DBR_r_arr,
    DBR_t_arr,
    DBR_phase_arr,
    DBR_r_at_target,
    DBR_phase_at_target,
    DBR_stopband_width,
    DBR_field_positions,
    DBR_field_values,
    DBR_n_field_arr,
    l_eff,
    interface_position,
) = analyze_DBR(DBR_flipped, target_wavelength, wavelength_arr)

# %%

"""
Calculate optical response.

"""

wavelength_arr = wavelength_arr_adaptive_mesh(
    target_wavelength - 100e-9, target_wavelength + 100e-9, target_wavelength, 5e-9
)

wavelength_arr, r_arr, t_arr, phase_arr = calculate_optical_properties(
    DBR, wavelength_arr
)

# %%

"""
Calculate electrical field distribution.

"""

field_positions, field_values, n_field_arr, Gamma_z, internal_loss = (
    analyze_electrical_field(DBR, target_wavelength)
)

# %% Reflectivity over N for even amount of layers

n1 = AlAs
n2 = GaAs

N_arr_even = []
R_arr_even = []

for N in range(25):
    N += 1
    DBR = build_DBR_structure(n1, n2, N, target_wavelength)
    wavelength_arr, r_arr, t_arr, phase_arr = calculate_optical_properties(
        DBR, wavelength_arr, Plot=False
    )
    N_arr_even.append(N)
    R_arr_even.append(max(r_arr))

plt.plot(N_arr_even, R_arr_even, label="Result", linestyle="", marker=".")
R_theory_even = [R_theoretical(N, n1, n2, 3.52, 1) for N in N_arr_even]
plt.plot(N_arr_even, R_theory_even, label="Theory")
plt.xlabel("Amount of mirror pairs N")
plt.ylabel("Reflectivity R")
plt.legend()

# %% Reflectivity over N for odd amount of layers

"""

Formula seems to be valid for big N only

"""

N_arr_odd = []
R_arr_odd = []

for N in range(25):
    N += 1.0  # to skip N = 0
    N += 0.5  # for uneven
    DBR = build_DBR_structure(2.95, 3.52, N, target_wavelength)
    wavelength_arr, r_arr, t_arr, phase_arr = calculate_optical_properties(
        DBR, wavelength_arr, Plot=False
    )
    N_arr_odd.append(N)
    R_arr_odd.append(max(r_arr))

plt.plot(N_arr_odd, R_arr_odd, label="Result", linestyle="", marker=".")
R_theory_odd = [R_theoretical(N, 2.95, 3.52, 3.52, 1) for N in N_arr_odd]
plt.plot(N_arr_odd, R_theory_odd, label="Theory")
plt.xlabel("Amount of mirror pairs N")
plt.ylabel("Reflectivity R")
plt.legend()

# %% Comparison

plt.plot(N_arr_even, R_arr_even, label="Result", linestyle="", marker=".")
plt.plot(N_arr_odd, R_arr_odd, linestyle="", marker=".")
plt.plot(N_arr_even, R_theory_even, label="Theory", color="tab:blue")
plt.plot(N_arr_odd, R_theory_odd, color="tab:orange")
plt.fill_between(
    N_arr_odd, R_arr_odd, R_arr_even, alpha=0.1, label="Accessible Reflectivities"
)

plt.xlabel("Amount of mirror pairs N")
plt.ylabel("Reflectivity R")
plt.legend()
plt.xlim(18, 25)
plt.ylim(0.97, 1.0)
# %%
