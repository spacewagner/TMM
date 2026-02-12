# %%

"""
Docstring for long_wavelength.lifetime_tuning

TODO

Evaluate different coating experiments. What is SiN instead of SiO, or reversed order of layers?
How thick should the SiN coating be, to match our experimental results?
Plot lifetime tuning for different DBR pairs, in the same plot.

"""

import matplotlib.pyplot as plt
import numpy as np

from TMM.structure_builder import (
    plot_structure,
    build_VCSEL_structure,
    build_DBR_structure,
    apply_AR_coating,
    apply_etch,
    build_active_region,
    VCSEL_embedding_active_region,
    get_VCSEL_structure,
)

from TMM.analysis import (
    analyse_VCSEL_lifetime_tuning,
    analyse_AR_coating,
    analyse_reflectivity_tuning,
    calculate_electrical_field,
    DBR_penetration_depth,
    structure_calculate_n_effective, analyse_electrical_field
)

from TMM.optics_utils import (
    refractive_index_AlGaAs_at_1310,
    refractive_index_SiO2,
    refractive_index_Si3N4,
    R_theoretical,
)

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
plt.title("Active Region")
plt.show()

d_optimum = 9.31953125e-07  # from cavity length optimization

VCSEL_active_region = VCSEL_embedding_active_region(VCSEL, active_region, d_optimum)
plot_structure(VCSEL_active_region)

get_VCSEL_structure(VCSEL_active_region)

results_electrical_field = calculate_electrical_field(
    VCSEL_active_region, target_wavelength, Plot=True
)

results_electrical_field = analyse_electrical_field(
    VCSEL_active_region, target_wavelength, Plot=True
)

results_electrical_field.Gamma_z_active_region
results_electrical_field.Gamma_z
# %% full embedded structure
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
active_region_idx = [5 + 2 * i for i in range(int(len(n_quantum_wells) / 2))]

# active region should be in center of embedding structure
d_expansion = np.sum(d_arr[: active_region_idx[0]]) - np.sum(
    d_arr[active_region_idx[-1] + 1 :]
)
d_arr.append(d_expansion)
n_arr.append(n_cavity)

embedding_full = build_active_region(n_arr, d_arr, active_region_idx=active_region_idx)
plot_structure(embedding_full)
plt.title("Active Region")
plt.show()

d_embedding_full = embedding_full["d"].sum()
delta_d = d_embedding_full - d_active_region
d_optimum = d_optimum - delta_d

VCSEL_embedding_full = VCSEL_embedding_active_region(VCSEL, embedding_full, d_optimum)
plot_structure(VCSEL_embedding_full)

results_electrical_field = analyse_electrical_field(
    VCSEL_embedding_full, target_wavelength, Plot=True
)

VCSEL_structure_interpolated = results_electrical_field.structure_interpolated
# get cavity indices for passive and active VCSEL structures for later calculation of effective cavity length and effective refractive index
idx_cavity = VCSEL_structure_interpolated.loc[
    VCSEL_structure_interpolated["name"] == "Cavity"
].index

idx_embedding = VCSEL_structure_interpolated.loc[
    VCSEL_structure_interpolated["name"] == "Embedding"
].index

if not idx_embedding.empty:
    idx_cavity = idx_embedding

# filter cavity structure
cavity = VCSEL_structure_interpolated.iloc[idx_cavity[0] : idx_cavity[-1]]

# define start and stop of cavity
pos_start = cavity["position"].values[0]
pos_stop = cavity["position"].values[-1] + cavity["d"].values[-1]

n_effective = structure_calculate_n_effective(
    results_electrical_field, pos_start, pos_stop
)
print(n_effective)
# %%
VCSEL = apply_AR_coating(VCSEL, refractive_index_Si3N4(1310e-9), 470e-9)
# VCSEL = apply_AR_coating(VCSEL, refractive_index_SiO2(1310e-9), 200e-9)

plot_structure(VCSEL)
plt.title("Long-Wavelength VCSEL Structure")
# %% Lifetime Tuning

VCSEL = VCSEL_embedding_full
VCSEL = apply_AR_coating(VCSEL, refractive_index_Si3N4(target_wavelength), 470e-9)
VCSEL_cavity_tuning_properties = analyse_VCSEL_lifetime_tuning(
    VCSEL,
    target_wavelength,
    n_coating=refractive_index_SiO2(target_wavelength),
    resolution=10e-9,
    Plot=True,
)
VCSEL_cavity_tuning_properties.Gamma_z
print(VCSEL_cavity_tuning_properties.v_gr_coating_arr.min())
print(VCSEL_cavity_tuning_properties.v_gr_etch_arr.min())
# %% Lifetime Tuning for different DBR configurations

N_top_arr = [21, 21.5, 22]
for N in N_top_arr:

    VCSEL_temp = build_VCSEL_structure(
        n_bottom_1,
        n_bottom_2,
        N_bottom,
        n_top_1,
        n_top_2,
        N,
        n_cavity,
        n_substrate,
        n_air,
        target_wavelength,
        N_cavity=5,
    )

    VCSEL_cavity_tuning_properties = analyse_VCSEL_lifetime_tuning(
        VCSEL_temp,
        target_wavelength,
        n_coating=refractive_index_SiO2(target_wavelength),
        resolution=2e-9,
        Plot=True,
    )
    plt.show()


# %% Build top DBR

DBR_top = build_DBR_structure(
    n_top_1, n_top_2, N_top, target_wavelength, n_cavity, n_air
)
plot_structure(DBR_top)
plt.show()
analyse_reflectivity_tuning(DBR_top, target_wavelength)
plt.show()

# compare with Si3N4 coating
DBR_top_coated = apply_AR_coating(DBR_top, refractive_index_Si3N4(1310e-9), 470e-9)
plot_structure(DBR_top_coated)
plt.show()
analyse_reflectivity_tuning(DBR_top_coated, target_wavelength)
plt.show()

# %% How does the tuning rage of R vary with the coating material?
n_coating_arr = np.linspace(1, 6, 15)
R_tuning_range_arr = []
for n in n_coating_arr:
    analyse_AR_coating_results = analyse_AR_coating(
        DBR_top, target_wavelength, n, Plot=True
    )
    R_tuning_range_arr.append(analyse_AR_coating_results.R_tuning_range)
plt.xlim(0, 300)
plt.margins(0, 0)
# plt.ylim(top=1)
plt.show()

plt.plot(n_coating_arr, R_tuning_range_arr)
plt.xlabel("Coating Refractive Index n")
plt.ylabel("DBR Reflectivity Tuning Range")
plt.show()
# %% How does tuning range change with the Si3N4 coating underneath the SiO2 coating?

Si3N4_thickness_arr = np.linspace(0, 600e-9, 30)
start_reflectivity_arr = []
R_tuning_range_arr = []
for SiN_thickness in Si3N4_thickness_arr:
    DBR_top_temp = apply_AR_coating(
        DBR_top, refractive_index_Si3N4(target_wavelength), SiN_thickness
    )
    analyse_AR_coating_results = analyse_AR_coating(
        DBR_top_temp,
        target_wavelength,
        refractive_index_SiO2(target_wavelength),
        Plot=True,
    )
    start_reflectivity_arr.append(analyse_AR_coating_results.R_arr[0])
    R_tuning_range_arr.append(analyse_AR_coating_results.R_tuning_range)
plt.show()

plt.plot(start_reflectivity_arr, R_tuning_range_arr, marker="", linestyle="-")
plt.xlabel("Reflectivity without SiO2 Coating")
plt.ylabel("DBR Reflectivity Tuning Range")
plt.show()
plt.plot(Si3N4_thickness_arr, R_tuning_range_arr)
plt.xlabel("Si3N4 Coating thickness (nm)")
plt.ylabel("DBR Reflectivity Tuning Range")
plt.show()
 # %% How does tuning range change with the DBR layers underneath the SiO2 coating?

d_etch_arr = np.linspace(0, 600e-9, 50)
R_tuning_range_arr = []
start_reflectivity_arr = []
for d_etch in d_etch_arr:
    DBR_top_temp = apply_etch(DBR_top, d_etch)
    analyse_AR_coating_results = analyse_AR_coating(
        DBR_top_temp,
        target_wavelength,
        refractive_index_SiO2(target_wavelength),
        Plot=True,
    )
    start_reflectivity_arr.append(analyse_AR_coating_results.R_arr[0])
    R_tuning_range_arr.append(analyse_AR_coating_results.R_tuning_range)
plt.show()

plt.plot(start_reflectivity_arr, R_tuning_range_arr, marker="", linestyle="-")
plt.xlabel("Reflectivity without SiO2 Coating")
plt.ylabel("DBR Reflectivity Tuning Range")
plt.show()
plt.plot(d_etch_arr, R_tuning_range_arr)
plt.xlabel("Etch thickness (nm)")
plt.ylabel("DBR Reflectivity Tuning Range")
plt.show()
# %%
