# %%
import matplotlib.pyplot as plt
import numpy as np
import locale
from matplotlib.colors import Normalize

from TMM.structure_builder import (
    plot_structure,
    build_VCSEL_structure,
    build_DBR_structure,
    apply_AR_coating,
    build_active_region,
    VCSEL_embedding_active_region,
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

from TMM.optimizations import plot_VCSEL_embedding_sweep, optimize_embedding_thickness

# def apply_plot_style():
#     plt.minorticks_on()
#     plt.grid(which="major", linestyle="-", linewidth=0.4)
#     plt.tick_params(direction="in", which="both", top=True, right=True)
#     # plt.margins(x=0, y=0)
#     plt.gcf().set_size_inches(8.5, 6)


# def apply_latex_style_us(final_output=False):
#     try:
#         locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
#     except locale.Error:
#         print("Warning: Could not set locale to 'en_US.UTF-8'.")

#     plt.rcParams["text.usetex"] = final_output
#     plt.rcParams["font.family"] = ["Latin Modern Roman"]

#     if final_output:
#         plt.rcParams["text.latex.preamble"] = (
#             r"\usepackage["
#             r"locale=US,"
#             r"per-mode=symbol,"
#             r"separate-uncertainty,"
#             r"sticky-per,"
#             r"group-digits=false,"
#             r"detect-all"
#             r"]{siunitx}"
#             r"\usepackage[T1]{fontenc}"
#             r"\usepackage{lmodern}"
#             r"\usepackage{microtype}"
#         )

#     plt.rcParams["axes.formatter.use_locale"] = not final_output
#     plt.rcParams["font.size"] = 14
# apply_latex_style_us()

# plt.style.use("style.mplstyle")

# plt.rcParams["text.latex.preamble"] = [
#     r"\usepackage[locale=US,per-mode=symbol,separate-uncertainty,sticky-per,group-digits=false,detect-all]{siunitx}",
#     r"\usepackage[T1]{fontenc}",
#     r"\usepackage{lmodern}",
#     r"\usepackage{microtype}",
# ]
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


analyse_VCSELs_DBRs(VCSEL, target_wavelength, wavelength_arr)
# %% Mirror Reflectivity

"""

Compare etching properties from TMM with analytical solution

"""

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

    # --- annotations stay on ax1 ---
    arrowprops = dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=90,rad=0")

    plt.annotate(
        "This work", xytext=(20.2, 0.85), xy=(21.5, 0.99), arrowprops=arrowprops
    )

    plt.xlabel(r"Amount of mirror pairs $N$")
    plt.ylabel(r"Reflectance $R$")
    plt.legend()
    plt.ylim(0, 1)
    plt.xlim(0, 25.5)
    plt.savefig("mirror_reflectivity_over_N.pdf", bbox_inches="tight", dpi=600)
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
        "This work", xytext=(890, 0.07), xy=(1009.23125, 0.04), arrowprops=arrowprops
    )

    plt.xlim(0, 1200)
    plt.ylim(0, 0.25)
    plt.xlabel("Cavity Length (nm)")
    plt.ylabel("Longitudinal Confinement $\\Gamma_{\\mathrm{z}}$")
    plt.legend()
    plt.savefig("cavity_length_optimization.pdf", dpi=600, bbox_inches="tight")
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
        arrowstyle="->", connectionstyle="angle,angleA=0,angleB=90,rad=0", linewidth=0.5
    )

    ax1.annotate(
        "FL",
        xytext=(4.5, 3.0),
        xy=(4.43407605882437, 3.2),
        arrowprops=arrowprops,
        fontsize=fontsize,
    )
    ax1.annotate(
        "FL",
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
        arrowprops=dict(arrowstyle="<-", linewidth=0.5),
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
        arrowprops=dict(arrowstyle="<-", linewidth=0.5),
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
        arrowprops=dict(arrowstyle="<->", linewidth=0.5),
        fontsize=fontsize,
    )
    ax1.text(4.63, 3.60, "Top n-InP IC", ha="center", va="bottom", fontsize=fontsize)

    ax1.annotate(
        "",
        xytext=(4.97725168382436, 3.58),
        xy=(5.43942715257435, 3.58),
        arrowprops=dict(arrowstyle="<->", linewidth=0.5),
        fontsize=fontsize,
    )
    ax1.text(5.21, 3.60, "Bottom n-InP IC", ha="center", va="bottom", fontsize=fontsize)

    ax1.annotate(
        "",
        xytext=(4.43407605882437, 3.75),
        xy=(5.43942715257435, 3.75),
        arrowprops=dict(arrowstyle="<->", linewidth=0.5),
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
    plt.savefig("1310_mode_profile.pdf", bbox_inches="tight", dpi=600)
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
VCSEL_cavity_tuning_properties = analyse_VCSEL_lifetime_tuning(
    VCSEL_passivated,
    target_wavelength,
    n_coating=refractive_index_SiO2(target_wavelength),
    resolution=10e-9,
    Plot=True,
    alpha_i_arr=[0, 500, 1000, 2000],
)

plt.savefig("VCSEL_tuning_analysis.pdf", bbox_inches="tight", dpi=600)
plt.show()
VCSEL_cavity_tuning_properties.Gamma_z
print(VCSEL_cavity_tuning_properties.v_gr_coating_arr.min())
print(VCSEL_cavity_tuning_properties.v_gr_etch_arr.min())
print(VCSEL_cavity_tuning_properties.Gamma_z)
print(VCSEL_cavity_tuning_properties.g_threshold_coating_arr_arr[2])

# %%
from matplotlib.patches import Ellipse


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
        ax1.axvline(0, color="tab:red")

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
        ax3.axvline(0, color="tab:red")

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
            65,
            "SiO$_2$ \n Coating",
            ha="center",
            va="bottom",
        )
        ax3.text(
            -235,
            65,
            "Si$_3$N$_4$ \n Passivation",
            ha="center",
            va="bottom",
        )
        ax3.text(
            -668,
            65,
            "AlGaAs/GaAs \n Top DBR",
            ha="center",
            va="bottom",
        )

        ax2.set_ylabel(r"Photon lifetime $\tau_{\mathrm{ph}}$ (ps)")
        ax1.set_ylabel(r"Reflectance $R$")
        ax4.set_ylabel(r"Threshold gain $g_{\mathrm{th}}$ (cm$^{-1}$)")
        ax3.set_ylabel(r"Mirror loss $\alpha_\mathrm{m}$ (cm$^{-1}$)")
        ax3.set_xlabel(r"Thickness (nm)")

        # ax1.minorticks_on()
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
            bbox_to_anchor=(0.51, 0.495),  # x,y in figure coordinates
            ncol=4,
            frameon=True,
        )


plot_analyse_VCSEL_lifetime_tuning(VCSEL_cavity_tuning_properties)
plt.savefig("VCSEL_tuning_analysis.pdf", bbox_inches="tight", dpi=600)
plt.show()
# %%
