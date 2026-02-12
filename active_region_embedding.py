# %%

"""
TODO

structure based on Sergey A. Blokhin et al. “High Power Single Mode 1300-Nm Superlattice Based VCSEL: Impact of the Buried Tunnel Junction Diameter on Performance” but for 1310nm instead 1300nm

add a 470nm Si3N4 passivation layer

"""

import numpy as np
import matplotlib.pyplot as plt

from TMM.structure_builder import (
    plot_structure,
    apply_AR_coating,
    build_VCSEL_structure,
    wavelength_arr_adaptive_mesh,
    build_active_region,
)
from TMM.field_solver import (
    calculate_optical_properties,
    calculate_electrical_field,
)

from TMM.analysis import (
    analyse_VCSEL,
    analyse_electrical_field,
    analyse_VCSEL_lifetime_tuning,
    VCSEL_embedding_active_region,
)

from TMM.optics_utils import (
    refractive_index_AlGaAs_at_1310,
    refractive_index_SiO2,
    refractive_index_Si3N4,
)

from TMM.optimizations import plot_VCSEL_embedding_sweep, optimize_embedding_thickness

from TMM.outputs import plot_analyse_VCSEL_lifetime_tuning

# from my_utils.plot_settings import apply_plot_style
# from my_utils.latex_settings import apply_latex_style_us
# apply_latex_style_us(final_output=True)

# %%
"""
Build and plot a VCSEL Structure.

"""

# 1310
GaAs = refractive_index_AlGaAs_at_1310(0)
Al91Ga9As = refractive_index_AlGaAs_at_1310(91)

n_bottom_1 = GaAs
n_bottom_2 = Al91Ga9As
N_bottom = 35.5
n_top_1 = GaAs
n_top_2 = Al91Ga9As
N_top = 21.5
n_cavity = 3.205
n_substrate = GaAs
n_air = 1
target_wavelength = 1310e-9

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

# VCSEL = apply_AR_coating(VCSEL, refractive_index_Si3N4(1310e-9), 470e-9)
# VCSEL = apply_AR_coating(VCSEL, refractive_index_SiO2(1310e-9), 200e-9)

plot_structure(VCSEL)
plt.title("Long-Wavelength VCSEL Structure")

# %% Full Analysis

analyse_VCSEL_results = analyse_VCSEL(VCSEL, target_wavelength)
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
plt.figure(figsize=(10, 6))
R_arr = VCSEL_optical_properties_result.R_arr
T_arr = VCSEL_optical_properties_result.T_arr
phase_arr = VCSEL_optical_properties_result.phase_arr
plt.plot(wavelength_arr * 1e9, R_arr, label="Reflection")
plt.plot(wavelength_arr * 1e9, T_arr, label="Transmission", alpha=0.6)
plt.plot(wavelength_arr * 1e9, R_arr + T_arr, label="R+T", alpha=0.6)
plt.xlabel("Wavelength (nm)")
plt.legend()
# apply_plot_style()

# %%
"""
analyse the electrical field distribution within the VCSEL.

"""

VCSEL_field_properties_results = analyse_electrical_field(VCSEL, target_wavelength)
plt.figure(figsize=(10, 6))
plot_structure(VCSEL)
plt.plot(
    np.array(VCSEL_field_properties_results.field_positions_arr) * 1e6,
    abs(VCSEL_field_properties_results.field_values_arr) ** 2
    / np.max(abs(VCSEL_field_properties_results.field_values_arr) ** 2)
    * np.max(np.real(VCSEL_field_properties_results.n_field_arr)),
    color="tab:red",
    label="norm. $|E|^2$",
)
plt.title("")
plt.show()

# %% Define active region

n_quantum_wells = [3.49, 3.53] * 24
d_quantum_wells = [2e-9, 0.8e-9] * 24
n_quantum_wells.append(3.49)
d_quantum_wells.append(2e-9)

n_arr = [3.49, *n_quantum_wells, 3.49]
d_arr = [4e-9, *d_quantum_wells, 4e-9]
active_region_idx = [2 + 2 * i for i in range(int(len(n_quantum_wells) / 2))]

active_region = build_active_region(n_arr, d_arr, active_region_idx=active_region_idx)
d_active_region = active_region["d"].sum()
plot_structure(active_region)
# plt.axvline(x=d_active_region / 2 * 1e6, color="red", linestyle="--")
plt.title("Active Region")
plt.show()

# %%

results = optimize_embedding_thickness(
    VCSEL,
    active_region,
    target_wavelength=target_wavelength,
    d_min=0e-9,
    d_max=1050e-9,
)

# %% plotting
plot_VCSEL_embedding_sweep(results, plot_embedding_thickness=False)
plt.xlim(0, 1200)
plt.ylim(0, 1)
plt.ylabel(r"$\Gamma_{\mathrm{z}}$")
plt.xlabel("Cavity Length (nm)")
# apply_plot_style()
plt.gcf().set_size_inches(8, 6)
# plt.savefig("cavity_length_optimization.pdf", dpi=600, bbox_inches="tight")
# %% Build VCSEL with active region, with optimum cavity length

VCSEL_modified = VCSEL_embedding_active_region(
    VCSEL, active_region, results.d_optimum_arr[4]
)

results_electrical_field = calculate_electrical_field(
    VCSEL_modified, target_wavelength, Plot=False
)
plt.plot(
    results_electrical_field.field_positions_arr * 1e6,
    np.abs(results_electrical_field.field_values_arr) ** 2
    / np.max(np.abs(results_electrical_field.field_values_arr) ** 2)
    * np.max(results_electrical_field.n_field_arr),
    label="norm. |E^2|",
)
plt.plot(
    results_electrical_field.field_positions_arr * 1e6,
    results_electrical_field.n_field_arr,
)
# plt.xlim(8.75, 9.0)
plt.xlabel("$Position~(\\mu m)$")
plt.ylabel("Refractive index")
# apply_plot_style()

# %% embed the whole structure

n_arr = [
    3.53,
    3.49,
    3.24,
    3.49,
    *n_quantum_wells,
    3.49,
    3.205,
    3.53,
]
d_arr = [30e-9, 12e-9, 36e-9, 4e-9, *d_quantum_wells, 4e-9, 50e-9, 10e-9]
active_region_idx = [4 + i for i in range(len(n_quantum_wells))]

# active region should be in center of embedding structure
d_expansion = np.sum(d_arr[: active_region_idx[0]]) - np.sum(
    d_arr[active_region_idx[-1] + 1 :]
)
d_arr.append(d_expansion)
n_arr.append(n_cavity)

embedding_full = build_active_region(n_arr, d_arr, active_region_idx=active_region_idx)

d_embedding_full = embedding_full["d"].sum()
plot_structure(embedding_full)
plt.axvline(x=d_embedding_full / 2 * 1e6, color="red", linestyle="--")
plt.title("Full Embedded Structure")
plt.show()

delta_d = d_embedding_full - d_active_region
d_optimum = results.d_optimum_arr[4] - delta_d

# should be equal
d_cavity = results.d_optimum_arr[4] + d_active_region
d_cavity = d_optimum + d_embedding_full

d_cavity_blokhin = 990e-9  # +-3nm, extracted with imageJ
print(
    f"Cavity length @ {target_wavelength*1e9}nm = {d_cavity*1e9:.3f}nm \nBlokhin @1300nm = {d_cavity_blokhin*1e9:.3f}nm"
)

VCSEL_modified2 = VCSEL_embedding_active_region(VCSEL, embedding_full, d_optimum)

results_electrical_field = calculate_electrical_field(
    VCSEL_modified2, target_wavelength, Plot=False
)

plt.plot(
    results_electrical_field.field_positions_arr * 1e6,
    np.abs(results_electrical_field.field_values_arr) ** 2
    / np.max(np.abs(results_electrical_field.field_values_arr) ** 2)
    * np.max(results_electrical_field.n_field_arr),
    label="norm. |E^2|",
)

plt.plot(
    results_electrical_field.field_positions_arr * 1e6,
    results_electrical_field.n_field_arr,
)
# apply_plot_style()

# %% compare results from active region only, with full embedded structure

VCSEL_modified = VCSEL_embedding_active_region(
    VCSEL, active_region, results.d_optimum_arr[4]
)

plot_structure(VCSEL_modified)
plot_structure(VCSEL_modified2)

from TMM.analysis import analyse_electrical_field

analyse_electrical_field(VCSEL_modified, target_wavelength)

results_electrical_field = calculate_electrical_field(
    VCSEL_modified, target_wavelength, Plot=False
)

plt.plot(
    results_electrical_field.field_positions_arr * 1e6,
    np.abs(results_electrical_field.field_values_arr) ** 2
    / np.max(np.abs(results_electrical_field.field_values_arr) ** 2)
    * np.max(results_electrical_field.n_field_arr),
    label="without BTJ",
)

results_electrical_field = calculate_electrical_field(
    VCSEL_modified2, target_wavelength, Plot=False
)

plt.plot(
    results_electrical_field.field_positions_arr * 1e6,
    np.abs(results_electrical_field.field_values_arr) ** 2
    / np.max(np.abs(results_electrical_field.field_values_arr) ** 2)
    * np.max(results_electrical_field.n_field_arr),
    linestyle="--",
    label="with BTJ",
)

plt.xlim(7.5, 9.75)
plt.xlabel("$Position~(\\mu m)$")
plt.ylabel("Refractive index")
plt.legend()
# apply_plot_style()

# %%

VCSEL_modified2 = VCSEL_embedding_active_region(VCSEL, embedding_full, d_optimum)

plot_structure(VCSEL_modified2)

results_electrical_field = calculate_electrical_field(
    VCSEL_modified2, target_wavelength, Plot=False
)

plt.plot(
    results_electrical_field.field_positions_arr * 1e6,
    np.abs(results_electrical_field.field_values_arr) ** 2
    / np.max(np.abs(results_electrical_field.field_values_arr) ** 2)
    * np.max(results_electrical_field.n_field_arr),
    label=r"norm. $|\mathrm{E}|^2$",
    color="tab:red",
)

plt.xlabel(r"Position ($\mu$m)")
plt.ylabel("Refractive index")
plt.legend(loc="lower right")
# apply_plot_style()
plt.ylim(ymin=0, ymax=3.6)
plt.xlim(7.95, 9.7)
plt.gcf().set_size_inches((8, 6))
plt.title("")
# plt.savefig("1310_mode_profile.pdf", dpi=600, bbox_inches="tight")
# %%
position_resolution = 100
for i, d_opt in enumerate(results.d_optimum_arr):

    if i in [0, 1, 4]:

        VCSEL_modified = VCSEL_embedding_active_region(VCSEL, active_region, d_opt)

        idx_embedding = VCSEL_modified.loc[VCSEL_modified["name"] == "Embedding"].index

        results_electrical_field = calculate_electrical_field(
            VCSEL_modified, target_wavelength, Plot=False
        )
        x = results_electrical_field.field_positions_arr[
            idx_embedding[0]
            * position_resolution : (idx_embedding[1] + 1)
            * position_resolution
        ]
        x = x - np.min(x)
        x = x / np.max(x)
        plt.plot(
            x,
            np.abs(
                results_electrical_field.field_values_arr[
                    idx_embedding[0]
                    * position_resolution : (idx_embedding[1] + 1)
                    * position_resolution
                ]
            )
            ** 2
            / np.max(np.abs(results_electrical_field.field_values_arr) ** 2),
            label=f"N={i+1}",
        )
plt.legend()
plt.xlabel("Cavity length (a.u.)")
plt.ylabel(r"norm. $|\mathrm{E}|^2$")
# apply_plot_style()
plt.gcf().set_size_inches(8, 6)
plt.show()


# %% Lifetime Tuning

n_coating = refractive_index_SiO2(target_wavelength)
VCSEL_cavity_tuning_properties = analyse_VCSEL_lifetime_tuning(
    VCSEL,
    target_wavelength,
    n_coating=n_coating,
    resolution=2e-9,
    Plot=False,
)
plot_analyse_VCSEL_lifetime_tuning(VCSEL_cavity_tuning_properties)

# %%
