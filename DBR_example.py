# %%

"""
TODO

Compare to examples from Chuang and Coldren


"""

import matplotlib.pyplot as plt
import numpy as np

from TMM.structure_builder import (
    build_DBR_structure,
    plot_structure,
    wavelength_arr_adaptive_mesh,
    flip_structure,
)
from TMM.field_solver import calculate_optical_properties
from TMM.optics_utils import R_theoretical

from TMM.analysis import (
    analyse_electrical_field,
    analyse_DBR,
    analyse_reflectivity_tuning,
    analyse_etching,
)

from TMM.optics_utils import refractive_index_AlGaAs_at_940, refractive_index_SiO2


# %%

"""

Define DBR structure

"""
target_wavelength = 940e-9
GaAs = refractive_index_AlGaAs_at_940(0)
AlAs = refractive_index_AlGaAs_at_940(100)

n1 = GaAs
n2 = AlAs
n_substrate = 3.3
n_air = 1
N = 21.5

# Set wavelength range for the spectra by using the adaptive mesh function
wavelength_arr = wavelength_arr_adaptive_mesh(
    target_wavelength - 100e-9, target_wavelength + 100e-9, target_wavelength, 5e-9
)

# Build DBR structure
DBR = build_DBR_structure(
    n1, n2, N, target_wavelength, n_substrate=n_substrate, n_air=n_air
)
# Plot the DBR structure
plot_structure(DBR)

# %%
# run full analysis on the DBR
DBR_results = analyse_DBR(DBR, target_wavelength, wavelength_arr)

# %%

"""
Flip the structure to imitate incident wave from the opposite side.

"""

DBR_flipped = flip_structure(DBR)
plot_structure(DBR_flipped)
plt.show()

DBR_flipped_results = analyse_DBR(DBR_flipped, target_wavelength, wavelength_arr)

# %% Reflectivity tuning
n_coating = refractive_index_SiO2(target_wavelength)
analyse_reflectivity_tuning(DBR, target_wavelength, n_coating=n_coating)

# %%

"""
Calculate optical response.

"""

wavelength_arr = wavelength_arr_adaptive_mesh(
    target_wavelength - 100e-9, target_wavelength + 100e-9, target_wavelength, 5e-9
)

DBR_optical_properties_result = calculate_optical_properties(DBR, wavelength_arr)

# %%

"""
Calculate electrical field distribution.

"""

DBR_flipped_field_properties_results = analyse_electrical_field(
    DBR_flipped, target_wavelength
)

# %% Reflectivity over N for even amount of layers

n1 = AlAs
n2 = GaAs

N_arr_even = []
R_arr_even = []

for N in range(25):
    N += 1
    DBR = build_DBR_structure(n1, n2, N, target_wavelength)
    DBR_optical_properties_result = calculate_optical_properties(
        DBR, [target_wavelength], Plot=False
    )
    N_arr_even.append(N)
    R_arr_even.append(max(DBR_optical_properties_result.R_arr))

plt.plot(N_arr_even, R_arr_even, label="Result", linestyle="", marker=".")
R_theory_even = [R_theoretical(N, n1, n2, 3.52, 1) for N in N_arr_even]
plt.plot(N_arr_even, R_theory_even, label="Theory")
plt.xlabel("Amount of mirror pairs N")
plt.ylabel("Reflectivity R")
plt.legend()

# %% Reflectivity over N for odd amount of layers

N_arr_odd = []
R_arr_odd = []

for N in range(25):
    N += 1.0  # to skip N = 0
    N += 0.5  # for uneven
    DBR = build_DBR_structure(2.95, 3.52, N, target_wavelength)
    DBR_optical_properties_result = calculate_optical_properties(
        DBR, [target_wavelength], Plot=False
    )
    N_arr_odd.append(N)
    R_arr_odd.append(max(DBR_optical_properties_result.R_arr))

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

"""

Compare etching properties from TMM with analytical solution

"""
DBR_etching_properties = analyse_etching(DBR, target_wavelength, N=N)

plt.plot(DBR_etching_properties.d_arr, DBR_etching_properties.R_arr)
plt.show()
plt.plot(DBR_etching_properties.N_arr, DBR_etching_properties.R_arr)
plt.show()

# %%
N_arr_even = []
R_arr_even = []

for N in range(26):
    N += 0
    DBR = build_DBR_structure(
        n1, n2, N, target_wavelength, n_substrate=n_substrate, n_air=n_air
    )
    DBR_optical_properties_result = calculate_optical_properties(
        DBR, [target_wavelength], Plot=False
    )
    N_arr_even.append(N)
    R_arr_even.append(max(DBR_optical_properties_result.R_arr))

plt.plot(N_arr_even, R_arr_even, label="Result", linestyle="", marker=".")
R_theory_even = [R_theoretical(N, n1, n2, n_substrate, n_air) for N in N_arr_even]
plt.plot(N_arr_even, R_theory_even, label="Theory")
plt.xlabel("Amount of mirror pairs N")
plt.ylabel("Reflectivity R")
plt.legend()

# %% Reflectivity over N for odd amount of layers

N_arr_odd = []
R_arr_odd = []

for N in range(26):
    N += 0.0  # to skip N = 0
    N += 0.5  # for uneven
    DBR = build_DBR_structure(
        n1, n2, N, target_wavelength, n_substrate=n_substrate, n_air=n_air
    )
    DBR_optical_properties_result = calculate_optical_properties(
        DBR, [target_wavelength], Plot=False
    )
    N_arr_odd.append(N)
    R_arr_odd.append(max(DBR_optical_properties_result.R_arr))

plt.plot(N_arr_odd, R_arr_odd, label="Result", linestyle="", marker=".")
R_theory_odd = [R_theoretical(N, n1, n2, n_substrate, n_air) for N in N_arr_odd]
plt.plot(N_arr_odd, R_theory_odd, label="Theory")
plt.xlabel("Amount of mirror pairs N")
plt.ylabel("Reflectivity R")
plt.legend()

# %% Comparison
plt.plot(
    DBR_etching_properties.N_arr,
    DBR_etching_properties.R_arr,
    label="TMM Results",
)
plt.plot(N_arr_even, R_theory_even, marker=".", label="R(2N=even)")
plt.plot(N_arr_odd, R_theory_odd, marker=".", label="R(2N=odd)")


plt.xlabel("Amount of mirror pairs N")
plt.ylabel("Reflectivity R")
plt.legend()
plt.ylim(0, 1)
# %%

"""
Compare with examples from the literature
Example Chuang p.218 fig. 5.14

"""

target_wavelength = 980e-9
DBR = build_DBR_structure(3.52, 2.95, 20, target_wavelength, n_air=3.52)
DBR = DBR.iloc[1:-1]
plot_structure(DBR)

wavelength_arr = np.arange(820e-9, 1230e-9, 1e-9)
calculate_optical_properties(DBR, wavelength_arr)
# %%

"""
Compare with examples from the literature
Example Coldren p.120 fig. 3.14

"""

target_wavelength = 980e-9
n_high = 3.52  # GaAs @ 940nm chuang p. 218
n_low_arr = [2.88, 3.35, 3.45]  # AlAs @ 940nm
N_DBR_pairs = 20  # final amount of DBR pairs
wavelength_arr = np.arange(820e-9, 1230e-9, 1e-9)

r_arr_complete = []
t_arr_complete = []
phase_arr_complete = []

L_grating_arr = []

R_theory = []

for j, n_low in enumerate(n_low_arr):

    DBR = build_DBR_structure(n_high, n_low, 20, target_wavelength)
    DBR = DBR.iloc[1:-1]
    plot_structure(DBR)

    results = calculate_optical_properties(DBR, wavelength_arr, Plot=True)
    plt.show()

    r_arr_complete.append(results.R_arr)
    t_arr_complete.append(results.T_arr)
    phase_arr_complete.append(results.phase_arr)

    R_theory.append(R_theoretical(20, n_low, n_high, n_low, n_high))

    d_low = DBR.iloc[1]["d"]
    d_high = DBR.iloc[2]["d"]

    L_grating = (d_high + d_low) * N_DBR_pairs
    L_grating_arr.append(L_grating)


# %% Plot
def beta(wavelength):
    return 2 * np.pi / wavelength


delta = beta(target_wavelength) - beta(wavelength_arr)  # Coldren eq. 3.47


plt.plot(delta * L_grating_arr[0], r_arr_complete[0], label="r = 0.1")
plt.plot(delta * L_grating_arr[1], r_arr_complete[1], label="r = 0.25")
plt.plot(delta * L_grating_arr[2], r_arr_complete[2], label="r = 0.01")
plt.plot(
    [target_wavelength] * 3, R_theory, linestyle="", marker=".", label="R theoretical"
)
plt.xlabel("$\\delta L_{g} ~ (\\pi)$")
plt.ylabel("$|r_g|^2$")
plt.legend()
plt.show()

plt.plot(delta * L_grating_arr[0], phase_arr_complete[0], label="r = 0.1")
# plt.plot(delta * L_grating_arr[1], phase_arr_complete[1], label="r = 0.25")
plt.plot(delta * L_grating_arr[2], phase_arr_complete[2], label="r = 0.01")
plt.xlabel("$\\delta L_{g} ~ (\\pi)$")
plt.ylabel("$\\angle r_g$")
plt.legend()
plt.show()
# %%
