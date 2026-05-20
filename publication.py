# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from TMM.structure_builder import (
    plot_structure,
    build_VCSEL_structure,
    build_DBR_structure,
    apply_AR_coating,
    build_active_region,
    VCSEL_embedding_active_region,
    interpolate_structure,
)

from TMM.field_solver import calculate_optical_properties, calculate_electrical_field

from TMM.analysis import (
    analyse_etching,
    analyse_VCSEL_lifetime_tuning,
    VCSEL_embedding_active_region,
    analyse_VCSEL_lifetime_tuning,
    analyse_VCSELs_DBRs,
)

from TMM.optics_utils import (
    refractive_index_AlGaAs_at_1310,
    R_theoretical,
    refractive_index_SiO2,
    refractive_index_Si3N4,
)

from TMM.optimizations import optimize_embedding_thickness

# %% Build the VCSEL Structure
# structure approximated from Blokhin et. al.

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

with plt.style.context("style.mplstyle"):
    plot_structure(VCSEL)
    plt.title("Long-Wavelength VCSEL Structure")
    plt.show()

wavelength_arr = np.arange(target_wavelength - 100e-9, target_wavelength + 100e-9, 1e-9)
# %% Compare etching properties from TMM with analytical solution

n1 = GaAs
n2 = Al91Ga9As
n_substrate = n_cavity
n_air = 1

N_max = 25.5
DBR = build_DBR_structure(
    n1, n2, N_max, target_wavelength, n_substrate=n_substrate, n_air=n_air
)
plot_structure(DBR)
plt.show()
DBR_etching_properties = analyse_etching(DBR, target_wavelength, N=N_max)

with plt.style.context("style.mplstyle"):
    plt.plot(DBR_etching_properties.d_arr, DBR_etching_properties.R_arr)
    plt.show()
with plt.style.context("style.mplstyle"):
    plt.plot(DBR_etching_properties.N_arr, DBR_etching_properties.R_arr)
    plt.show()
# %%

# Reflectivity over N for even amount of layers

N_arr_even = []
R_arr_even = []

for N in range(int(N_max) + 1):
    DBR = build_DBR_structure(
        n1, n2, N, target_wavelength, n_substrate=n_substrate, n_air=n_air
    )
    DBR_optical_properties_result = calculate_optical_properties(
        DBR, [target_wavelength], Plot=False
    )
    N_arr_even.append(N)
    R_arr_even.append(max(DBR_optical_properties_result.R_arr))

R_theory_even = [R_theoretical(N, n1, n2, n_substrate, n_air) for N in N_arr_even]


# Reflectivity over N for odd amount of layers

N_arr_odd = []
R_arr_odd = []

for N in range(int(N_max) + 1):
    N += 0.5  # for uneven
    DBR = build_DBR_structure(
        n1, n2, N, target_wavelength, n_substrate=n_substrate, n_air=n_air
    )
    DBR_optical_properties_result = calculate_optical_properties(
        DBR, [target_wavelength], Plot=False
    )
    N_arr_odd.append(N)
    R_arr_odd.append(max(DBR_optical_properties_result.R_arr))

R_theory_odd = [R_theoretical(N, n1, n2, n_substrate, n_air) for N in N_arr_odd]
# Plotting

plt.plot(
    N_arr_even, R_arr_even, label="TMM, even", linestyle="", marker=".", color=f"C{0}"
)
plt.plot(N_arr_even, R_theory_even, label="Theory, even", color=f"C{0}")
plt.plot(
    N_arr_odd, R_arr_odd, label="TMM, odd", linestyle="", marker=".", color=f"C{1}"
)
plt.plot(N_arr_odd, R_theory_odd, label="Theory, odd", color=f"C{1}")
plt.xlabel("Amount of mirror pairs N")
plt.ylabel("Reflectivity R")
plt.legend()
plt.show()

# %% Comparison Plot

structure_interpolated = interpolate_structure(
    build_DBR_structure(
        n1, n2, 21.5, target_wavelength, n_substrate=n_substrate, n_air=n_air
    )
)
structure_interpolated["n"] = structure_interpolated["n"]

with plt.style.context("style.mplstyle"):

    plt.plot(DBR_etching_properties.N_arr, DBR_etching_properties.R_arr, label="TMM")

    plt.plot(
        N_arr_even,
        R_theory_even,
        marker=".",
        linestyle="",
        label="Analytical",
        color=f"C{1}",
    )
    plt.plot(
        N_arr_odd,
        R_theory_odd,
        marker=".",
        linestyle="",
        # label="R(2N=odd)",
        color=f"C{1}",
    )

    # plt.plot(
    #     N_arr_odd[-5],
    #     R_theory_odd[-5],
    #     marker=".",
    #     linestyle="",
    #     # label="R(2N=odd)",
    #     color=f"C{2}",
    # )

    # ax = plt.gca()
    # ax_inset = ax.inset_axes([0.55, 0.35, 0.4, 0.3])

    # ax_inset.plot(
    #     -1 * structure_interpolated["position"]
    #     + np.max(structure_interpolated["position"]) * 1e6,
    #     structure_interpolated["n"],
    #     label="$n(z)$ \n $N=21.5$",
    #     color=f"C{2}",
    # )
    # ax_inset.grid(False)
    # # ax_inset.tick_params(which = "both", top=False, right=False, bottom= True, left = True)
    # ax_inset.set_xticklabels([])
    # # ax_inset.set_yticklabels([])
    # # ax_inset.set_xlabel(r"Long. coordinate ($\mu m$)")
    # # ax_inset.set_ylabel("$n$")
    # ax_inset.legend(loc="lower right")

    # # --- annotations stay on ax1 ---
    # arrowprops = dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=90,rad=0")

    # plt.annotate("", xytext=(21.5, 0.6), xy=(21.5, 0.99), arrowprops=arrowprops)

    # --- annotations stay on ax1 ---
    arrowprops = dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=90,rad=0")

    plt.annotate(
        r"$N=21.5$",
        xytext=(21.5, 0.85),
        xy=(21.5, 0.99),
        arrowprops=arrowprops,
        ha="center",
        va="center",
    )

    plt.xlabel(r"Number of mirror pairs $N$")
    plt.ylabel(r"Reflectance $R$")
    plt.legend()
    plt.ylim(0, 1)
    plt.xlim(0, 25.5)
    plt.savefig("mirror_reflectivity_over_N.pdf", dpi=600)
    plt.show()
# %%


# %% Define active region

# structure approximated from Blokhin et. al.
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
# %%
d_min = 0e-9
d_max = 1050e-9
# d_min = 925e-9
# d_max = 940e-9

results = optimize_embedding_thickness(
    VCSEL,
    active_region,
    target_wavelength=target_wavelength,
    d_min=d_min,
    d_max=d_max,
)

# %% plot cavity length sweep and mode confinement

# plot_VCSEL_embedding_sweep(results, plot_embedding_thickness=False)
# plt.xlim(0, 1200)
# plt.ylim(0, 1)
# plt.ylabel(r"$\Gamma_{\mathrm{z}}$")
# plt.xlabel("Cavity Length (nm)")
# plt.savefig("cavity_length_optimization.pdf", dpi=600, bbox_inches="tight")
# plt.show()
with plt.style.context("style.mplstyle"):
    plt.plot(
        (results.d_embedding_arr + results.d_active_region) * 1e9,
        results.Gamma_z_active_region_arr,
        # label="Active Region",
    )
    i = 0
    for pos, val in zip(
        results.d_optimum_arr, results.Gamma_z_active_region_optimum_arr
    ):
        if i == 0:
            label = "Resonant Modes"
        else:
            label = None
        plt.plot(
            (pos + results.d_active_region) * 1e9,
            val,
            marker=".",
            linestyle="",
            color=f"C{1}",
            label=label,
        )
        i += 1

    # --- annotations stay on ax1 ---
    arrowprops = dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=90,rad=0")

    plt.annotate(
        r"0.5$\lambda$",
        xytext=(191.73125, 0.18331926 + 0.04),
        xy=(191.73125, 0.18331926 + 0.005),
        arrowprops=arrowprops,
        ha="center",
        va="bottom",
    )

    plt.annotate(
        r"2.5$\lambda$",
        xytext=(1009.23125, 0.03251929 + 0.04),
        xy=(1009.23125, 0.03251929 + 0.005),
        arrowprops=arrowprops,
        ha="center",
        va="bottom",
    )

    plt.xlim(0, 1200)
    plt.ylim(0, 0.25)
    plt.xlabel("Cavity Length $L$ (nm)")
    plt.ylabel("Longitudinal Confinement $\\Gamma_{\\mathrm{z}}$")
    plt.legend()
    plt.savefig("cavity_length_optimization.pdf", dpi=600)
    plt.show()

# %% Build and plot VCSEL with active region

peak_order = 4  # depending on the evaluated embedding range, it corresponds to long. mode order -1

VCSEL_modified = VCSEL_embedding_active_region(
    VCSEL, active_region, results.d_optimum_arr[peak_order]
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
plt.legend()
plt.xlabel("$Position~(\\mu m)$")
plt.ylabel("Refractive index")
plt.show()

# %% embed the whole structure, including In(Al)GaAs TJ, p-InAlAs, In(Al)GaAs-SL, n-InGaAs (Blokhin)

# structure approximated from Blokhin et. al.
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

# structure was rebuild from Blokhin et. al., but it is reversed to our used convetion of direction of light propagation, so it is reversed here
n_arr = n_arr[::-1]
d_arr = d_arr[::-1]

embedding_full = build_active_region(n_arr, d_arr, active_region_idx=active_region_idx)

d_embedding_full = embedding_full["d"].sum()
plot_structure(embedding_full)
plt.axvline(x=d_embedding_full / 2 * 1e6, color="red", linestyle="--")
plt.title("Full Embedded Structure")
plt.show()

# consistency check
delta_d = d_embedding_full - d_active_region
d_optimum = results.d_optimum_arr[peak_order] - delta_d
d_cavity = d_optimum + d_embedding_full
d_error = (results.d_optimum_arr[peak_order] + d_active_region) - (
    d_optimum + d_embedding_full
)
if d_error != 0:
    print("Embedding structure not of ideal thickness.")

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

# %%
total_length = np.max(results_electrical_field.field_positions_arr)
# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={"hspace": 0.0})

fontsize = 5

with plt.style.context("style.mplstyle"):
    ax1 = plt.subplot(2, 1, 1)

    # --- TOP: refractive index ---
    ax1.plot(
        -1 * (results_electrical_field.field_positions_arr - total_length) * 1e6,
        results_electrical_field.n_field_arr,
        color="C1",
    )

    ax1.set_ylabel(
        "Refractive index $n$",
        #    color=f"C{1}"
    )
    ax1.set_ylim(2.6, 4.0)

    # --- annotations stay on ax1 ---
    arrowprops = dict(
        arrowstyle="->",
        connectionstyle="angle,angleA=0,angleB=90,rad=0",
        linewidth=0.5,
        shrinkA=0,
        shrinkB=0,
    )

    ax1.annotate(
        "FI",
        xytext=(4.5, 3.0),
        xy=(4.43407605882437, 3.2),
        arrowprops=arrowprops,
        fontsize=fontsize,
    )
    ax1.annotate(
        "FI",
        xytext=(5.29, 3.0),
        xy=(5.43942715257435, 3.2),
        arrowprops=arrowprops,
        fontsize=fontsize,
    )
    ax1.annotate(
        "In(Al)GaAs TJ",
        xytext=(4.4, 2.85),
        xy=(4.84303168382437, 3.2),
        arrowprops=arrowprops,
        fontsize=fontsize,
    )
    ax1.annotate(
        "p-InAlAs",
        xytext=(4.47, 2.7),
        xy=(4.88281168382437, 3.2),
        arrowprops=arrowprops,
        fontsize=fontsize,
    )
    ax1.annotate(
        "In(Al)GaAs SL",
        xytext=(5.1, 2.7),
        xy=(4.935, 3.2),
        arrowprops=arrowprops,
        fontsize=fontsize,
    )
    ax1.annotate(
        "n-InGaAs",
        xytext=(5.145, 2.85),
        xy=(5.03219168382436, 3.2),
        arrowprops=arrowprops,
        fontsize=fontsize,
    )

    ax1.annotate(
        "",
        xytext=(4, 3.466),
        xy=(4.43407605882437, 3.466),
        arrowprops=dict(arrowstyle="<-", linewidth=0.5, shrinkA=0, shrinkB=0),
        fontsize=fontsize,
    )
    ax1.text(
        4.2,
        3.486,
        "Top DBR \n AlGaAs/GaAs",
        ha="center",
        va="bottom",
        fontsize=fontsize,
    )

    ax1.annotate(
        "",
        xytext=(6, 3.466),
        xy=(5.43942715257435, 3.466),
        arrowprops=dict(arrowstyle="<-", linewidth=0.5, shrinkA=0, shrinkB=0),
        fontsize=fontsize,
    )
    ax1.text(
        5.7,
        3.486,
        "Bottom DBR \n AlGaAs/GaAs",
        ha="center",
        va="bottom",
        fontsize=fontsize,
    )

    ax1.annotate(
        "",
        xytext=(4.43407605882437, 3.58),
        xy=(4.82209168382437, 3.58),
        arrowprops=dict(arrowstyle="<->", linewidth=0.5, shrinkA=0, shrinkB=0),
        fontsize=fontsize,
    )
    ax1.text(4.63, 3.60, "Top n-InP IC", ha="center", va="bottom", fontsize=fontsize)

    ax1.annotate(
        "",
        xytext=(4.97725168382436, 3.58),
        xy=(5.43942715257435, 3.58),
        arrowprops=dict(arrowstyle="<->", linewidth=0.5, shrinkA=0, shrinkB=0),
        fontsize=fontsize,
    )
    ax1.text(5.21, 3.60, "Bottom n-InP IC", ha="center", va="bottom", fontsize=fontsize)

    ax1.annotate(
        "",
        xytext=(4.43407605882437, 3.75),
        xy=(5.43942715257435, 3.75),
        arrowprops=dict(arrowstyle="<->", linewidth=0.5, shrinkA=0, shrinkB=0),
        fontsize=fontsize,
    )
    ax1.text(
        4.95,
        3.77,
        "InAlGaAsP Optical Cavity",
        ha="center",
        va="bottom",
        fontsize=fontsize,
    )

    ax2 = plt.subplot(2, 1, 2)

    # --- BOTTOM: field intensity ---
    ax2.plot(
        -1 * (results_electrical_field.field_positions_arr - total_length) * 1e6,
        np.abs(results_electrical_field.field_values_arr) ** 2
        / np.max(np.abs(results_electrical_field.field_values_arr) ** 2),
        color="C0",
    )

    ax2.set_ylabel(
        "$|E|^2$ (a.u.)"
        #    , color=f"C{0}"
    )
    ax2.set_xlabel("Longitudinal coordinate (µm)")
    ax2.set_ylim(0, 1.0)

    # --- cosmetics: make it look like one box ---
    # ax1.spines["bottom"].set_visible(False)
    # ax2.spines["top"].set_visible(False)

    ax1.tick_params(labelbottom=False)  # hide x labels on top plot
    ax1.set_xlim(3.95, 6.03)
    ax2.set_xlim(3.95, 6.03)
    ax1.set_xlim(4, 6)
    ax2.set_xlim(4, 6)
    # ax1.minorticks_on()
    # ax1.grid(which="major", linestyle="-", linewidth=0.4)
    # ax1.tick_params(
    #     direction="in", which="both", top=True, right=True, bottom=True, labelbottom=False
    # )
    # ax2.tick_params(
    #     direction="in", which="both", top=True, right=True, bottom=True, labelbottom=True
    # )
    plt.savefig("1310_mode_profile.pdf", dpi=600)
    plt.show()

# %%

VCSEL_passivated = apply_AR_coating(
    VCSEL_modified2, refractive_index_Si3N4(1310e-9), 470e-9
)

# VCSEL_passivated = apply_AR_coating(
#     VCSEL_modified2, refractive_index_SiO2(1310e-9), 470e-9
# )

plot_structure(VCSEL_passivated)
plt.title("Long-Wavelength VCSEL Structure")
plt.show()
# %%

# TODO setze gruppengeschwindigkeit manuell, prüfe confinement faktor mit publikation 0.047

# internal_loss_arr = [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
internal_loss_arr = np.linspace(0, 2000, 50)
# internal_loss_arr = [0, 500, 1000, 2000]

VCSEL_cavity_tuning_properties = analyse_VCSEL_lifetime_tuning(
    VCSEL_passivated,
    target_wavelength,
    n_coating=refractive_index_SiO2(target_wavelength),
    resolution=1e-9,
    Plot=True,
    alpha_i_arr=internal_loss_arr,
)

# plt.savefig("VCSEL_tuning_analysis.pdf", dpi=600)
plt.show()
VCSEL_cavity_tuning_properties.Gamma_z
print(VCSEL_cavity_tuning_properties.v_gr_coating_arr.min())
print(VCSEL_cavity_tuning_properties.v_gr_etch_arr.min())
print(VCSEL_cavity_tuning_properties.Gamma_z)
print(VCSEL_cavity_tuning_properties.g_threshold_coating_arr_arr[2])

# %%
d_arr = np.concatenate(
    [
        np.flip(-1 * VCSEL_cavity_tuning_properties.d_etching_arr),
        VCSEL_cavity_tuning_properties.d_coating_arr,
    ]
)

R_arr = np.concatenate(
    [
        np.flip(VCSEL_cavity_tuning_properties.R_etching_arr),
        VCSEL_cavity_tuning_properties.R_coating_arr,
    ]
)

n_arr = np.concatenate(
    [
        np.flip(VCSEL_cavity_tuning_properties.n_etching_arr),
        VCSEL_cavity_tuning_properties.n_coating_arr,
    ]
)

alpha_m_arr = np.concatenate(
    [
        np.flip(VCSEL_cavity_tuning_properties.alpha_m_etch_arr),
        VCSEL_cavity_tuning_properties.alpha_m_coating_arr,
    ]
)
i = 1
print(f"internal loss: {VCSEL_cavity_tuning_properties.alpha_i_arr[i]}")
photon_lifetime_arr = np.concatenate(
    [
        np.flip(VCSEL_cavity_tuning_properties.photon_lifetime_etch_arr_arr[i]),
        VCSEL_cavity_tuning_properties.photon_lifetime_coating_arr_arr[i],
    ]
)

g_threshold_arr = np.concatenate(
    [
        np.flip(VCSEL_cavity_tuning_properties.g_threshold_etch_arr_arr[i]),
        VCSEL_cavity_tuning_properties.g_threshold_coating_arr_arr[i],
    ]
)

n_unique_arr = np.unique(n_arr)
d_mask = d_arr >= -600e-9
d_arr_filtered = d_arr[d_mask]
n_arr_filtered = n_arr[d_mask]
R_arr_filtered = R_arr[d_mask]
photon_lifetime_arr_filtered = photon_lifetime_arr[d_mask]
alpha_m_arr_filtered = alpha_m_arr[d_mask]
g_threshold_arr_filtered = g_threshold_arr[d_mask]

for n_unique in n_unique_arr:
    n_mask = n_arr_filtered == n_unique
    d_arr_filtered_masked = d_arr_filtered[n_mask]
    n_arr_filtered_masked = n_arr_filtered[n_mask]
    R_arr_filtered_masked = R_arr_filtered[n_mask]
    photon_lifetime_arr_filtered_masked = photon_lifetime_arr_filtered[n_mask]
    alpha_m_arr_filtered_masked = alpha_m_arr_filtered[n_mask]
    g_threshold_arr_filtered_masked = g_threshold_arr_filtered[n_mask]
    plt.plot(n_arr_filtered_masked, R_arr_filtered_masked)

    print("\n")

    print(
        f"{n_unique}, d for max={d_arr_filtered_masked[np.argmax(R_arr_filtered_masked)]}, d for min={d_arr_filtered_masked[np.argmin(R_arr_filtered_masked)]}, delta d = {np.abs(d_arr_filtered_masked[np.argmax(R_arr_filtered_masked)]-d_arr_filtered_masked[np.argmin(R_arr_filtered_masked)])}"
    )

    print(
        f"{n_unique}, Rmax={np.max(R_arr_filtered_masked)}, Rmin={np.min(R_arr_filtered_masked)}, deltaR = {np.abs(np.max(R_arr_filtered_masked)-np.min(R_arr_filtered_masked))}"
    )

    print(
        f"{n_unique}, tau_max={np.max(photon_lifetime_arr_filtered_masked)}, tau_min={np.min(photon_lifetime_arr_filtered_masked)}, delta_tau = {np.abs(np.max(photon_lifetime_arr_filtered_masked)-np.min(photon_lifetime_arr_filtered_masked))}"
    )

    print(
        f"{n_unique}, alpha_m_max={np.max(alpha_m_arr_filtered_masked)}, alpha_m_min={np.min(alpha_m_arr_filtered_masked)}, delta_alpha_m = {np.abs(np.max(alpha_m_arr_filtered_masked)-np.min(alpha_m_arr_filtered_masked))}"
    )

    print(
        f"{n_unique}, g_th_max={np.max(g_threshold_arr_filtered_masked)}, g_th_min={np.min(g_threshold_arr_filtered_masked)}, delta_g_th = {np.abs(np.max(g_threshold_arr_filtered_masked)-np.min(g_threshold_arr_filtered_masked))}"
    )
# %%
d_roi_arr = [0, 120e-9]

print(f"internal loss: {VCSEL_cavity_tuning_properties.alpha_i_arr[i]}")

for d_roi in d_roi_arr:
    n_unique_arr = np.unique(n_arr)
    d_mask = d_arr == d_arr[np.argmin(np.abs(d_arr - d_roi))]
    n_arr_filtered = n_arr[d_mask]
    R_arr_filtered = R_arr[d_mask]
    photon_lifetime_arr_filtered = photon_lifetime_arr[d_mask]
    alpha_m_arr_filtered = alpha_m_arr[d_mask]
    g_threshold_arr_filtered = g_threshold_arr[d_mask]

    print("\n\n")
    print(
        f"Thickness: {d_roi} - Reflectance: {R_arr_filtered[0]} - Mirror Loss: {alpha_m_arr_filtered[0]} - Photon Lifetime: {photon_lifetime_arr_filtered[0]} - Threshold Gain: {g_threshold_arr_filtered[0]}"
    )

# %%
import pandas as pd

records = []

for i, alpha_i in enumerate(internal_loss_arr):

    photon_lifetime = np.concatenate(
        [
            np.flip(VCSEL_cavity_tuning_properties.photon_lifetime_etch_arr_arr[i]),
            VCSEL_cavity_tuning_properties.photon_lifetime_coating_arr_arr[i],
        ]
    )

    g_threshold = np.concatenate(
        [
            np.flip(VCSEL_cavity_tuning_properties.g_threshold_etch_arr_arr[i]),
            VCSEL_cavity_tuning_properties.g_threshold_coating_arr_arr[i],
        ]
    )

    temp_df = pd.DataFrame(
        {
            "alpha_i": alpha_i,
            "d": d_arr,
            "n": n_arr,
            "R": R_arr,
            "tau_p": photon_lifetime,
            "alpha_m": alpha_m_arr,
            "g_th": g_threshold,
        }
    )

    records.append(temp_df)

df = pd.concat(records, ignore_index=True)
# %%
d_mask = d_arr == d_arr[np.argmin(np.abs(d_arr - d_roi))]

d_roi_arr = [0, 39e-9, 120e-9, 2.65e-07]

# cmap = plt.get_cmap("viridis")
# norm = plt.Normalize(vmin=min(d_roi_arr), vmax=max(d_roi_arr))

colors = [f"C{0}", f"C{2}", f"C{1}", f"C{3}"]

with plt.style.context("style.mplstyle"):
    fig, ax = plt.subplots()
    for i, d_roi in enumerate(d_roi_arr):

        # nearest available thickness
        nearest_d = df.loc[(df["d"] - d_roi).abs().idxmin(), "d"]

        df_roi = df[df["d"] == nearest_d]

        if d_roi not in {0, 120e-9}:
            linestyle = "--"
        else:
            linestyle = "-"

        ax.plot(
            df_roi["alpha_i"] * 1e-2,
            df_roi["tau_p"] * 1e12,
            label=f"{np.abs(nearest_d) * 1e9:.0f}$~$nm",
            linestyle=linestyle,
            color=colors[i],
            # color=cmap(norm(d_roi))
        )

    d_roi = 39e-9
    nearest_d = df.loc[(df["d"] - d_roi).abs().idxmin(), "d"]
    df_roi_min = df[df["d"] == nearest_d]
    d_roi = 265e-9
    nearest_d = df.loc[(df["d"] - d_roi).abs().idxmin(), "d"]
    df_roi_max = df[df["d"] == nearest_d]

    plt.fill_between(
        df_roi_min["alpha_i"] * 1e-2,
        df_roi_min["tau_p"] * 1e12,
        df_roi_max["tau_p"] * 1e12,
        alpha=0.2,
        color="grey",
        #  label = "Intermediate"
    )

    # sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    # fig.colorbar(sm, ax=ax, label="Thickness Modification (nm)")

    plt.xlabel(r"Internal Loss $\alpha_i$ (cm$^{-1}$)")
    plt.ylabel(r"Photon Lifetime $\tau_p$ (ps)")
    plt.xlim(0, 20)
    plt.ylim(0, 12)
    plt.legend()
    plt.savefig("photon_lifetime_over_internal_loss.pdf", dpi=600)
    plt.show()

# %%
internal_loss_arr = [0, 500, 1000, 2000]

VCSEL_cavity_tuning_properties = analyse_VCSEL_lifetime_tuning(
    VCSEL_passivated,
    target_wavelength,
    n_coating=refractive_index_SiO2(target_wavelength),
    resolution=1e-9,
    Plot=True,
    alpha_i_arr=internal_loss_arr,
)

# plt.savefig("VCSEL_tuning_analysis.pdf", dpi=600)
plt.show()
# %% with inset

from matplotlib.ticker import MultipleLocator


def plot_analyse_VCSEL_lifetime_tuning(VCSEL_cavity_tuning_properties, Save_to="pdf"):

    d_arr = np.concatenate(
        [
            np.flip(-1 * VCSEL_cavity_tuning_properties.d_etching_arr),
            VCSEL_cavity_tuning_properties.d_coating_arr,
        ]
    )

    R_arr = np.concatenate(
        [
            np.flip(VCSEL_cavity_tuning_properties.R_etching_arr),
            VCSEL_cavity_tuning_properties.R_coating_arr,
        ]
    )

    n_arr = np.concatenate(
        [
            np.flip(VCSEL_cavity_tuning_properties.n_etching_arr),
            VCSEL_cavity_tuning_properties.n_coating_arr,
        ]
    )

    with plt.style.context("style.mplstyle"):

        plt.figure(figsize=(3.34645667 * 2, 2.36220472441 * 2))
        ax1 = plt.subplot(2, 1, 1)  # Span column 1 of row 1

        ax1.plot(d_arr * 1e9, R_arr, color="black")

        # ax_inset = ax1.inset_axes([1 - 0.3375, 0.45, 0.3, 0.3])

        # for d_roi in d_roi_arr:

        #     # nearest available thickness
        #     nearest_d = df.loc[(df["d"] - d_roi).abs().idxmin(), "d"]

        #     df_roi = df[df["d"] == nearest_d]

        #     ax_inset.plot(
        #         df_roi["alpha_i"] * 1e-2,
        #         df_roi["tau_p"] * 1e12,
        #         label=f"{np.abs(nearest_d) * 1e9:.0f}$~$nm",
        #     )

        # ax_inset.text(6, 4, "0 nm", ha="center", va="center", color=f"C{0}")
        # ax_inset.text(6, 6.5, "120 nm", ha="center", va="center", color=f"C{1}")

        # ax_inset.set_xlabel(r"$\alpha_i$ (cm$^{-1}$)")
        # ax_inset.set_ylabel(r"$\tau_p$ (ps)")
        # ax_inset.set_xlim(0, 20)
        # ax_inset.set_ylim(2.5, 7.5)

        ax2 = ax1.twinx()
        ax2.grid(False)

        for i, alpha_i in enumerate(VCSEL_cavity_tuning_properties.alpha_i_arr):
            photon_lifetime_arr = np.concatenate(
                [
                    np.flip(
                        VCSEL_cavity_tuning_properties.photon_lifetime_etch_arr_arr[i]
                    ),
                    VCSEL_cavity_tuning_properties.photon_lifetime_coating_arr_arr[i],
                ]
            )
            ax2.plot(
                d_arr * 1e9,
                photon_lifetime_arr * 1e12,
                linestyle="--",
                label=f"$\\alpha_i$ = {alpha_i*1e-2:.1f}" " cm$^{-1}$",
                color=f"C{i}",
            )

        ax3 = plt.subplot(2, 1, 2)

        alpha_m_arr = np.concatenate(
            [
                np.flip(VCSEL_cavity_tuning_properties.alpha_m_etch_arr),
                VCSEL_cavity_tuning_properties.alpha_m_coating_arr,
            ]
        )
        ax3.plot(d_arr * 1e9, alpha_m_arr * 1e-2, color="black")

        ax4 = ax3.twinx()
        ax4.grid(False)
        for i, alpha_i in enumerate(VCSEL_cavity_tuning_properties.alpha_i_arr):
            g_threshold_arr = np.concatenate(
                [
                    np.flip(VCSEL_cavity_tuning_properties.g_threshold_etch_arr_arr[i]),
                    VCSEL_cavity_tuning_properties.g_threshold_coating_arr_arr[i],
                ]
            )
            ax4.plot(
                d_arr * 1e9,
                g_threshold_arr * 1e-2,
                linestyle="--",
                label=f"$\\alpha_{{\\mathrm{{i}}}}$ = {alpha_i*1e-2:.1f}"
                " cm$^{{-1}}$",
                color=f"C{i}",
            )

        ax1.axvline(0, color="black", linestyle=":")
        ax3.axvline(0, color="black", linestyle=":")
        ax1.axvline(120, linestyle=":", color=f"C{3}")
        ax3.axvline(120, linestyle=":", color=f"C{3}")

        # norm = Normalize(vmin=n_arr.min(), vmax=n_arr.max())
        norm = Normalize(vmin=1, vmax=n_arr.max())
        C = np.vstack([n_arr, n_arr])

        # ax1
        X, Y = np.meshgrid(d_arr * 1e9, [np.min(R_arr), np.max(R_arr)])
        pcm = ax1.pcolormesh(
            X,
            Y,
            C,
            shading="auto",
            cmap="viridis_r",
            norm=norm,
            alpha=0.3,
            rasterized=True,
        )

        # # inset
        # X, Y = np.meshgrid(d_arr * 1e9, [2.5, 7.5])
        # pcm = ax_inset.pcolormesh(
        #     X,
        #     Y,
        #     C,
        #     shading="auto",
        #     cmap="viridis_r",
        #     norm=norm,
        #     alpha=0.3,
        #     rasterized=True,
        # )

        # ax2
        X, Y = np.meshgrid(
            d_arr * 1e9, [np.min(alpha_m_arr * 1e-2), np.max(alpha_m_arr * 1e-2)]
        )
        pcm = ax3.pcolormesh(
            X,
            Y,
            C,
            shading="auto",
            cmap="viridis_r",
            norm=norm,
            alpha=0.3,
            rasterized=True,
        )

        # Draw a circle/ellipse around a group of lines
        ellipse = Ellipse(
            xy=(0, 4.75),  # center in data coordinates
            width=100,  # in data units
            height=6.25,
            angle=0,
            fill=False,
            edgecolor="black",
            linewidth=1,
            linestyle="--",
            transform=ax2.transData,  # or ax2.transData for the right axis
        )
        ax2.add_patch(ellipse)

        # Arrow pointing to the axis label (or anywhere)
        ax2.annotate(
            "",
            xy=(100, 7.875),  # arrow tip (target)
            xytext=(-5, 7.875),  # arrow tail (start, e.g. edge of ellipse)
            arrowprops=dict(arrowstyle="->", color="black"),
        )

        # Draw a circle/ellipse around a group of lines
        ellipse = Ellipse(
            xy=(0, 880),  # center in data coordinates
            width=100,  # in data units
            height=1000,
            angle=0,
            fill=False,
            edgecolor="black",
            linewidth=1,
            linestyle="--",
            transform=ax4.transData,  # or ax2.transData for the right axis
        )
        ax4.add_patch(ellipse)

        # Arrow pointing to the axis label (or anywhere)
        ax4.annotate(
            "",
            xy=(100, 1380),  # arrow tip (target)
            xytext=(-5, 1380),  # arrow tail (start, e.g. edge of ellipse)
            arrowprops=dict(arrowstyle="->", color="black"),
        )

        ax3.annotate(
            "",
            xy=(-880, 70),  # arrow tip (target)
            xytext=(-775, 70),  # arrow tail (start, e.g. edge of ellipse)
            arrowprops=dict(arrowstyle="->", color="black"),
        )

        ax1.annotate(
            "",
            xy=(-880, 0.986),  # arrow tip (target)
            xytext=(-803, 0.986),  # arrow tail (start, e.g. edge of ellipse)
            arrowprops=dict(arrowstyle="->", color="black"),
        )

        ax3.text(
            225,
            70,
            "SiO$_2$ \n $n=1.45$",
            ha="center",
            va="center",
        )
        ax3.text(
            -235,
            70,
            "Si$\\textsubscript{x}$N$\\textsubscript{y}$ \n $n=2.00$",
            ha="center",
            va="center",
        )
        # ax3.text(
        #     -580,
        #     70,
        #     "Top DBR",
        #     ha="center",
        #     va="center",
        # )

        ax3.text(
            -575,
            70,
            "AlGaAs \n $n=2.96$",
            ha="right",
            va="center",
        )

        ax3.text(
            -550,
            70,
            "GaAs \n $n=3.41$",
            ha="left",
            va="center",
        )

        ax1.text(
            19,
            1.002,
            "Etching $\\leftarrow \\rightarrow$ Deposition",
            ha="center",
            va="center",
        )

        ax2.set_ylabel(r"Photon lifetime $\tau_{\mathrm{ph}}$ (ps)")
        ax1.set_ylabel(r"Top DBR Reflectance $R$")
        ax4.set_ylabel(r"Threshold gain $g_{\mathrm{th}}$ (cm$^{-1}$)")
        ax3.set_ylabel(r"Mirror loss $\alpha_\mathrm{m}$ (cm$^{-1}$)")
        ax3.set_xlabel("Thickness Modification (nm)")

        ax1.minorticks_on()
        # ax2.minorticks_on()
        # ax3.minorticks_on()
        # ax4.minorticks_on()

        # ax1.tick_params(direction="in", which="both", top=True)
        # ax1.grid(which="major", linestyle="-", linewidth=0.5)
        ax1.tick_params(axis="x", labelbottom=False)

        # ax2.tick_params(direction="in", which="both")

        # ax3.tick_params(direction="in", which="both", top=True)
        # ax3.grid(which="major", linestyle="-", linewidth=0.5)
        # ax3.autoscale(enable=True, axis="x", tight=True)
        # ax3.autoscale(enable=True, axis="y", tight=True)

        # ax4.tick_params(direction="in", which="both")

        ax3.set_ylim(0, 80)
        ax4.set_ylim(0, 4000)
        ax1.set_ylim(0.95, 1)
        ax2.set_ylim(0, 25)
        ax1.set_xlim(xmax=450)
        ax3.set_xlim(xmax=450)
        ax1.yaxis.set_major_locator(MultipleLocator(0.01))
        # Get all handles and labels
        handles3, labels3 = ax3.get_legend_handles_labels()
        handles4, labels4 = ax4.get_legend_handles_labels()
        all_handles = handles3 + handles4
        all_labels = labels3 + labels4
        # ax4.legend(all_handles, all_labels, loc="best", ncol=2)

        fig = plt.gcf()

        fig.legend(
            all_handles,
            all_labels,
            loc="center",
            bbox_to_anchor=(0.513, 0.4945),  # x,y in figure coordinates
            ncol=4,
            frameon=True,
        )


plot_analyse_VCSEL_lifetime_tuning(VCSEL_cavity_tuning_properties)
plt.savefig("VCSEL_tuning_analysis.pdf", dpi=600)
plt.show()
# %% plot of tau over peak order and internal loss

peak_order_arr = [
    0,
    1,
    2,
    3,
    4,
]  # depending on the evaluated embedding range, it corresponds to long. mode order -1

for peak_order in peak_order_arr:

    VCSEL_modified = VCSEL_embedding_active_region(
        VCSEL, active_region, results.d_optimum_arr[peak_order]
    )

    # consistency check
    delta_d = d_embedding_full - d_active_region
    d_optimum = results.d_optimum_arr[peak_order] - delta_d

    VCSEL_modified2 = VCSEL_embedding_active_region(VCSEL, embedding_full, d_optimum)

    VCSEL_passivated = apply_AR_coating(
        VCSEL_modified2, refractive_index_Si3N4(1310e-9), 470e-9
    )

    # internal_loss_arr = np.linspace(0, 2000, 100)
    internal_loss_arr = [0, 500, 1000, 2000]

    VCSEL_cavity_tuning_properties = analyse_VCSEL_lifetime_tuning(
        VCSEL_passivated,
        target_wavelength,
        n_coating=refractive_index_SiO2(target_wavelength),
        resolution=10e-9,
        Plot=True,
        alpha_i_arr=internal_loss_arr,
    )

    # plt.savefig("VCSEL_tuning_analysis.pdf", dpi=600)
    plt.show()
    VCSEL_cavity_tuning_properties.Gamma_z

    d_arr = np.concatenate(
        [
            np.flip(-1 * VCSEL_cavity_tuning_properties.d_etching_arr),
            VCSEL_cavity_tuning_properties.d_coating_arr,
        ]
    )

    R_arr = np.concatenate(
        [
            np.flip(VCSEL_cavity_tuning_properties.R_etching_arr),
            VCSEL_cavity_tuning_properties.R_coating_arr,
        ]
    )

    n_arr = np.concatenate(
        [
            np.flip(VCSEL_cavity_tuning_properties.n_etching_arr),
            VCSEL_cavity_tuning_properties.n_coating_arr,
        ]
    )

    records = []

    for i, alpha_i in enumerate(internal_loss_arr):

        photon_lifetime = np.concatenate(
            [
                np.flip(VCSEL_cavity_tuning_properties.photon_lifetime_etch_arr_arr[i]),
                VCSEL_cavity_tuning_properties.photon_lifetime_coating_arr_arr[i],
            ]
        )

        g_threshold = np.concatenate(
            [
                np.flip(VCSEL_cavity_tuning_properties.g_threshold_etch_arr_arr[i]),
                VCSEL_cavity_tuning_properties.g_threshold_coating_arr_arr[i],
            ]
        )

        temp_df = pd.DataFrame(
            {
                "alpha_i": alpha_i,
                "d": d_arr,
                "n": n_arr,
                "R": R_arr,
                "tau_p": photon_lifetime,
                "alpha_m": alpha_m_arr,
                "g_th": g_threshold,
            }
        )

        records.append(temp_df)

    df = pd.concat(records, ignore_index=True)

    d_roi_arr = [0, 120e-9]

    with plt.style.context("style.mplstyle"):
        fig, ax = plt.subplots()
        for d_roi in d_roi_arr:

            # nearest available thickness
            nearest_d = df.loc[(df["d"] - d_roi).abs().idxmin(), "d"]

            df_roi = df[df["d"] == nearest_d]

            ax.plot(
                df_roi["alpha_i"] * 1e-2,
                df_roi["tau_p"] * 1e12,
                label=f"{np.abs(nearest_d) * 1e9:.0f}$~$nm",
            )

    plt.xlabel(r"Internal Loss $\alpha_i$ (cm$^{-1}$)")
    plt.ylabel(r"Photon Lifetime $\tau_p$ (ps)")
    plt.xlim(0, 30)
    plt.ylim(0, 8)
    plt.legend()
    plt.show()
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Parameters
# ============================================================

peak_order_arr = [0, 1, 2, 3, 4]
internal_loss_arr = [0, 500, 1000, 2000]
d_roi_arr = [0, 120e-9]

all_records = []


# ============================================================
# Main simulation loop
# ============================================================

for peak_order in peak_order_arr:

    # --------------------------------------------------------
    # Build VCSEL structure
    # --------------------------------------------------------

    VCSEL_modified = VCSEL_embedding_active_region(
        VCSEL,
        active_region,
        results.d_optimum_arr[peak_order],
    )

    delta_d = d_embedding_full - d_active_region
    d_optimum = results.d_optimum_arr[peak_order] - delta_d

    VCSEL_modified = VCSEL_embedding_active_region(
        VCSEL,
        embedding_full,
        d_optimum,
    )

    VCSEL_passivated = apply_AR_coating(
        VCSEL_modified,
        refractive_index_Si3N4(1310e-9),
        470e-9,
    )

    # --------------------------------------------------------
    # Run simulation
    # --------------------------------------------------------

    tuning = analyse_VCSEL_lifetime_tuning(
        VCSEL_passivated,
        target_wavelength,
        n_coating=refractive_index_SiO2(target_wavelength),
        resolution=10e-9,
        Plot=True,
        alpha_i_arr=internal_loss_arr,
    )

    plt.show()

    # --------------------------------------------------------
    # Common arrays
    # --------------------------------------------------------

    d_arr = np.concatenate(
        [
            np.flip(-1 * tuning.d_etching_arr),
            tuning.d_coating_arr,
        ]
    )

    R_arr = np.concatenate(
        [
            np.flip(tuning.R_etching_arr),
            tuning.R_coating_arr,
        ]
    )

    n_arr = np.concatenate(
        [
            np.flip(tuning.n_etching_arr),
            tuning.n_coating_arr,
        ]
    )

    alpha_m_arr_full = np.concatenate(
        [
            np.flip(tuning.alpha_m_etch_arr),
            tuning.alpha_m_coating_arr,
        ]
    )

    # --------------------------------------------------------
    # Internal loss loop
    # --------------------------------------------------------

    for i, alpha_i in enumerate(internal_loss_arr):

        tau_p_arr = np.concatenate(
            [
                np.flip(tuning.photon_lifetime_etch_arr_arr[i]),
                tuning.photon_lifetime_coating_arr_arr[i],
            ]
        )

        g_th_arr = np.concatenate(
            [
                np.flip(tuning.g_threshold_etch_arr_arr[i]),
                tuning.g_threshold_coating_arr_arr[i],
            ]
        )

        df_temp = pd.DataFrame(
            {
                "peak_order": peak_order,
                "alpha_i": alpha_i,
                "d": d_arr,
                "n": n_arr,
                "R": R_arr,
                "alpha_m": alpha_m_arr_full,
                "tau_p": tau_p_arr,
                "g_th": g_th_arr,
            }
        )

        all_records.append(df_temp)


# ============================================================
# Final master dataframe
# ============================================================

df = pd.concat(all_records, ignore_index=True)

print(df.head())
# %%
with plt.style.context("style.mplstyle"):

    for peak_order in peak_order_arr:

        fig, ax = plt.subplots()

        df_peak = df[df["peak_order"] == peak_order]

        for d_roi in d_roi_arr:

            nearest_d = df_peak.loc[(df_peak["d"] - d_roi).abs().idxmin(), "d"]

            df_roi = df_peak[df_peak["d"] == nearest_d]

            ax.plot(
                df_roi["alpha_i"] * 1e-2,
                df_roi["tau_p"] * 1e12,
                label=f"{abs(nearest_d)*1e9:.0f} nm",
            )

        ax.set(
            xlabel=r"Internal Loss $\alpha_i$ (cm$^{-1}$)",
            ylabel=r"Photon Lifetime $\tau_p$ (ps)",
            xlim=(0, 30),
            ylim=(0, 8),
        )

        ax.set_title(f"Peak Order {peak_order}")
        ax.legend()

        plt.show()
# %%
d_roi = 0e-9

# ------------------------------------------------------------
# Extract nearest d
# ------------------------------------------------------------

nearest_d = df.loc[
    (df["d"] - d_roi).abs().idxmin(),
    "d",
]

df_roi = df[df["d"] == nearest_d].copy()

# Unit conversions
df_roi["alpha_i_cm"] = df_roi["alpha_i"] * 1e-2
df_roi["tau_p_ps"] = df_roi["tau_p"] * 1e12

# ------------------------------------------------------------
# Pivot into 2D grid
# ------------------------------------------------------------

pivot = df_roi.pivot_table(
    index="peak_order",
    columns="alpha_i_cm",
    values="tau_p_ps",
)

X, Y = np.meshgrid(
    pivot.columns.values,
    pivot.index.values,
)

Z = pivot.values

from mpl_toolkits.mplot3d import Axes3D

with plt.style.context("style.mplstyle"):

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X,
        Y,
        Z,
        cmap="viridis",
        edgecolor="k",
        linewidth=0.3,
        alpha=0.9,
    )

    ax.set_xlabel(r"Internal Loss $\alpha_i$ (cm$^{-1}$)")
    ax.set_ylabel("Peak Order")
    ax.set_zlabel(r"Photon Lifetime $\tau_p$ (ps)")

    fig.colorbar(
        surf,
        shrink=0.7,
        label=r"$\tau_p$ (ps)",
    )

    plt.tight_layout()
    plt.show()

with plt.style.context("style.mplstyle"):

    fig, ax = plt.subplots()

    contour = ax.contourf(
        X,
        Y + 1,
        Z,
        levels=100,
        cmap="viridis",
    )

    cbar = fig.colorbar(
        contour,
        ax=ax,
        label=r"Photon Lifetime $\tau_p$ (ps)",
    )

    ax.set_xlabel(r"Internal Loss $\alpha_i$ (cm$^{-1}$)")
    ax.set_ylabel("Mode Order")

    plt.title(f"Photon Lifetime at d = {np.abs(nearest_d)*1e9:.0f} nm")
    plt.tight_layout()
    plt.show()
# %%
