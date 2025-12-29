# --- file: outputs.py ---
"""

This module contains functions for plotting and printing results from analysis.py


"""

import matplotlib.pyplot as plt
import time
import numpy as np
from TMM.structure_builder import interpolate_structure


def plot_analyse_VCSEL_lifetime_tuning(VCSEL_cavity_tuning_properties, Save_to="pdf"):

    VCSEL = VCSEL_cavity_tuning_properties.structure
    DBR_d1 = VCSEL.iloc[-2]["d"]
    DBR_d2 = VCSEL.iloc[-3]["d"]
    DBR_n1 = VCSEL.iloc[-2]["n"]
    DBR_n2 = VCSEL.iloc[-3]["n"]

    fig = plt.figure(figsize=(11, 8))

    ax1 = plt.subplot(3, 1, 1)  # Span column 1 of row 1

    ax1.plot(
        -1 * VCSEL_cavity_tuning_properties.d_etching_arr * 1e9,
        VCSEL_cavity_tuning_properties.R_etching_arr,
    )
    ax1.plot(
        VCSEL_cavity_tuning_properties.d_coating_arr * 1e9,
        VCSEL_cavity_tuning_properties.R_coating_arr,
        color="tab:blue",
    )
    ax1.axvline(0, color="tab:red")

    ax1.axvspan(
        VCSEL_cavity_tuning_properties.d_coating_arr[0] * 1e9,
        VCSEL_cavity_tuning_properties.d_coating_arr[-1] * 1e9,
        alpha=0.2,
        color="tab:green",
        label=VCSEL_cavity_tuning_properties.n_coating,
    )

    start_position = VCSEL_cavity_tuning_properties.d_etching_arr[0]
    label = None
    for i in range(2):

        end_position = start_position + DBR_d1
        if i == 1:
            label = DBR_n1
        ax1.axvspan(
            -1 * start_position * 1e9,
            -1 * end_position * 1e9,
            alpha=0.2,
            color="tab:blue",
            label=label,
        )
        start_position = end_position
        end_position = start_position + DBR_d2
        if i == 1:
            label = DBR_n2
        ax1.axvspan(
            -1 * start_position * 1e9,
            -1 * end_position * 1e9,
            alpha=0.2,
            color="tab:orange",
            label=label,
        )
        start_position = end_position

    ax1.minorticks_on()
    ax1.tick_params(direction="in", which="both", top=True)
    ax1.grid(which="both", linestyle="--", linewidth=0.5)
    ax1.minorticks_on()
    ax1.autoscale(enable=True, axis="x", tight=True)
    ax1.autoscale(enable=True, axis="y", tight=True)
    ax1.set_ylabel("Top DBR Reflectivity")
    ax1.tick_params(axis="x", labelbottom=False)

    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(
        -1 * VCSEL_cavity_tuning_properties.d_etching_arr * 1e9,
        VCSEL_cavity_tuning_properties.alpha_m_etch_arr * 1e-2,
    )
    ax2.plot(
        VCSEL_cavity_tuning_properties.d_coating_arr * 1e9,
        VCSEL_cavity_tuning_properties.alpha_m_coating_arr * 1e-2,
        color="tab:blue",
    )
    ax2.axvline(0, color="tab:red")

    ax2.axvspan(
        VCSEL_cavity_tuning_properties.d_coating_arr[0] * 1e9,
        VCSEL_cavity_tuning_properties.d_coating_arr[-1] * 1e9,
        alpha=0.2,
        color="tab:green",
        label=VCSEL_cavity_tuning_properties.n_coating,
    )

    start_position = VCSEL_cavity_tuning_properties.d_etching_arr[0]
    label = None
    for i in range(2):

        end_position = start_position + DBR_d1
        if i == 1:
            label = DBR_n1
        ax2.axvspan(
            -1 * start_position * 1e9,
            -1 * end_position * 1e9,
            alpha=0.2,
            color="tab:blue",
            label=label,
        )
        start_position = end_position
        end_position = start_position + DBR_d2
        if i == 1:
            label = DBR_n2
        ax2.axvspan(
            -1 * start_position * 1e9,
            -1 * end_position * 1e9,
            alpha=0.2,
            color="tab:orange",
            label=label,
        )
        start_position = end_position

    ax2.minorticks_on()
    ax2.tick_params(direction="in", which="both", top=True)
    ax2.grid(which="both", linestyle="--", linewidth=0.5)
    ax2.minorticks_on()
    ax2.autoscale(enable=True, axis="x", tight=True)
    ax2.autoscale(enable=True, axis="y", tight=True)
    ax2.set_ylabel("Mirror Loss (1/cm)")
    ax2.tick_params(axis="x", labelbottom=False)

    ax3 = plt.subplot(3, 1, 3)

    ax3.plot(
        -1 * VCSEL_cavity_tuning_properties.d_etching_arr * 1e9,
        VCSEL_cavity_tuning_properties.photon_lifetime_etch_arr * 1e12,
        label=f"$\\alpha_i$ = {VCSEL_cavity_tuning_properties.alpha_i*1e-2:.2}/cm",
    )
    ax3.plot(
        VCSEL_cavity_tuning_properties.d_coating_arr * 1e9,
        VCSEL_cavity_tuning_properties.photon_lifetime_coating_arr * 1e12,
        color="tab:blue",
    )

    ax3.axvline(0, color="tab:red")

    ax3.axvspan(
        VCSEL_cavity_tuning_properties.d_coating_arr[0] * 1e9,
        VCSEL_cavity_tuning_properties.d_coating_arr[-1] * 1e9,
        alpha=0.2,
        color="tab:green",
        label=VCSEL_cavity_tuning_properties.n_coating,
    )

    start_position = VCSEL_cavity_tuning_properties.d_etching_arr[0]
    label = None
    for i in range(2):

        end_position = start_position + DBR_d1
        if i == 1:
            label = DBR_n1
        ax3.axvspan(
            -1 * start_position * 1e9,
            -1 * end_position * 1e9,
            alpha=0.2,
            color="tab:blue",
            label=label,
        )
        start_position = end_position
        end_position = start_position + DBR_d2
        if i == 1:
            label = DBR_n2
        ax3.axvspan(
            -1 * start_position * 1e9,
            -1 * end_position * 1e9,
            alpha=0.2,
            color="tab:orange",
            label=label,
        )
        start_position = end_position

    ax3.minorticks_on()
    ax3.tick_params(direction="in", which="both", top=True)
    ax3.grid(which="both", linestyle="--", linewidth=0.5)
    ax3.minorticks_on()
    ax3.autoscale(enable=True, axis="x", tight=True)
    ax3.autoscale(enable=True, axis="y", tight=True)
    ax3.set_xlabel("Thickness (nm)")
    ax3.set_ylabel("Photon Lifetime (ps)")
    ax3.legend()

    if Save_to != None:
        formatted_time = time.strftime("%Y%m%d_%H%M%S")
        filename = f"VCSEL_tuning_analysis_{formatted_time}." + Save_to
        plt.savefig(filename, bbox_inches="tight", dpi=600)
        print("File saved to: " + filename)


def print_analyse_VCSEL_lifetime_tuning(VCSEL_cavity_tuning_properties):
    print("=" * 60 + "\nLifetime Tuning Analysis \n" + "=" * 60)
    print(
        f"R tuning range by AR coating: {VCSEL_cavity_tuning_properties.R_tuning_range_coating:.4f}"
    )
    print(
        f"R tuning range by etching: {VCSEL_cavity_tuning_properties.R_tuning_range_etching:.4f}"
    )
    print(
        f"Mirror loss tuning range by AR coating: {VCSEL_cavity_tuning_properties.alpha_m_coating_range*1e-2:.4f} /cm"
    )
    print(
        f"Mirror loss tuning range by etching: {VCSEL_cavity_tuning_properties.alpha_m_etch_range*1e-2:.4f} /cm"
    )
    print(
        f"Photon lifetime tuning range by AR coating: {VCSEL_cavity_tuning_properties.photon_lifetime_coating_range*1e12:.4f} ps"
    )
    print(
        f"Photon lifetime tuning range by etching: {VCSEL_cavity_tuning_properties.photon_lifetime_etch_range*1e12:.4f} ps"
    )


def plot_analyse_VCSEL(analyse_VCSEL_results, Save_to="pdf"):

    VCSEL = analyse_VCSEL_results.VCSEL

    # Create figure
    fig = plt.figure(figsize=(11, 8))

    # First row: full width plot (row 1, spans 2 columns)
    ax1 = plt.subplot(3, 2, (1, 2))  # Span columns 1-2 of row 1

    structure_interpolated = interpolate_structure(analyse_VCSEL_results.VCSEL)
    structure_interpolated["n"] = np.real(
        structure_interpolated["n"]
    )  # only plot Re(n)

    ax1.plot(
        structure_interpolated["position"] * 1e6,
        structure_interpolated["n"],
        label="n(z)",
    )

    cavity_start = float(
        structure_interpolated.loc[
            structure_interpolated["name"] == "Cavity", "position"
        ].iloc[0]
    )
    cavity_stop = float(
        structure_interpolated.loc[
            structure_interpolated["name"] == "Cavity", "position"
        ].iloc[-1]
    )

    n_max = VCSEL["n"].values.max()
    ax1.fill_between(
        np.array([cavity_start, cavity_stop]) * 1e6,
        [0, 0],
        [n_max, n_max],
        alpha=0.3,
        color="tab:red",
        label="Cavity",
    )
    ax1.plot(
        analyse_VCSEL_results.VCSEL_field_positions_arr * 1e6,
        abs(analyse_VCSEL_results.VCSEL_field_values_arr) ** 2
        / np.max(abs(analyse_VCSEL_results.VCSEL_field_values_arr) ** 2)
        * np.max(analyse_VCSEL_results.VCSEL_n_field_arr),
        color="tab:red",
        label="$|E|^2$",
    )

    # Combine them
    ax1.minorticks_on()
    ax1.tick_params(direction="in", which="both", top=True, right=True)
    ax1.grid(which="both", linestyle="--", linewidth=0.5)
    ax1.legend()
    ax1.autoscale(enable=True, axis="x", tight=True)
    ax1.autoscale(enable=True, axis="y", tight=True)
    ax1.set_xlabel("Postion $(\\mu m)$")
    ax1.set_ylabel("Refractive index")

    # Second row: two plots
    ax2 = plt.subplot(3, 2, 3)  # Row 2, Column 1
    ax1 = plt.gca()
    ax1.plot(
        analyse_VCSEL_results.wavelength_arr * 1e9,
        analyse_VCSEL_results.VCSEL_R_arr,
        color="tab:blue",
        label="Reflectivity",
    )
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Reflectivity", color="tab:blue")

    # Create second y-axis for phase
    ax2 = ax1.twinx()
    ax2.plot(
        analyse_VCSEL_results.wavelength_arr * 1e9,
        analyse_VCSEL_results.VCSEL_phase_arr / np.pi,
        color="tab:orange",
        alpha=0.5,
        label="Phase",
        linestyle=":",
    )

    ax1.minorticks_on()
    ax1.tick_params(direction="in", which="both", top=True)
    ax1.grid(which="both", linestyle="--", linewidth=0.5)
    ax2.set_ylabel("Phase (rad)", color="tab:orange")
    ax2.minorticks_on()
    ax2.tick_params(direction="in", which="both", top=True)
    # Add a single legend for both plots
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    ax2.set_ylim(-1, 1)

    if analyse_VCSEL_results.cavity_resonance_R <= 1:
        ax1.set_ylim(0, 1)
    ax1.autoscale(enable=True, axis="x", tight=True)

    ax3 = plt.subplot(3, 2, 4)  # Row 2, Column 2
    ax3.plot(
        analyse_VCSEL_results.cavity_wavelength_arr * 1e9,
        analyse_VCSEL_results.cavity_T_arr,
        label="T (leveled)",
    )
    ax3.axvline(
        analyse_VCSEL_results.cavity_resonance_wavelength * 1e9,
        linestyle=":",
        label=f"$\\lambda$: {analyse_VCSEL_results.cavity_resonance_wavelength*1e9:.1f}nm",
    )
    ax3.axhline(analyse_VCSEL_results.cavity_halfmax, linestyle=":")

    ax3.hlines(
        analyse_VCSEL_results.cavity_halfmax,
        analyse_VCSEL_results.cavity_hw_start * 1e9,
        analyse_VCSEL_results.cavity_hw_stop * 1e9,
        color="red",
        label=f"FWHM: {analyse_VCSEL_results.cavity_FWHM*1e9:.2f}nm"
        + "\n $\\tau_{p, limit}$:"
        + f"{analyse_VCSEL_results.cavity_photon_lifetime*1e12:.2f}ps",
    )

    ax3.minorticks_on()
    ax3.tick_params(direction="in", which="both", top=True, right=True)
    ax3.tick_params(axis="x", rotation=45)
    ax3.grid(which="both", linestyle="--", linewidth=0.5)
    ax3.set_xlabel("Wavelength (nm)")
    ax3.set_ylabel("Transmission")
    ax3.legend()
    ax3.autoscale(enable=True, axis="x", tight=True)
    ax3.autoscale(enable=True, axis="y", tight=True)

    # Third row: two plots
    ax4 = plt.subplot(3, 2, 5)  # Row 3, Column 1

    DBR_d1 = VCSEL.iloc[-2]["d"]
    DBR_d2 = VCSEL.iloc[-3]["d"]
    DBR_n1 = VCSEL.iloc[-2]["n"]
    DBR_n2 = VCSEL.iloc[-3]["n"]

    ax4.plot(
        -1 * analyse_VCSEL_results.d_etching_arr * 1e9,
        analyse_VCSEL_results.R_etching_arr,
    )
    ax4.plot(
        analyse_VCSEL_results.d_coating_arr * 1e9,
        analyse_VCSEL_results.R_coating_arr,
        color="tab:blue",
    )

    ax4.axvline(0, color="tab:red")

    ax4.axvspan(
        analyse_VCSEL_results.d_coating_arr[0] * 1e9,
        analyse_VCSEL_results.d_coating_arr[-1] * 1e9,
        alpha=0.2,
        color="tab:green",
        label=analyse_VCSEL_results.n_coating,
    )

    start_position = analyse_VCSEL_results.d_etching_arr[0]
    label = None
    for i in range(2):

        end_position = start_position + DBR_d1
        if i == 1:
            label = DBR_n1
        ax4.axvspan(
            -1 * start_position * 1e9,
            -1 * end_position * 1e9,
            alpha=0.2,
            color="tab:blue",
            label=label,
        )
        start_position = end_position
        end_position = start_position + DBR_d2
        if i == 1:
            label = DBR_n2
        ax4.axvspan(
            -1 * start_position * 1e9,
            -1 * end_position * 1e9,
            alpha=0.2,
            color="tab:orange",
            label=label,
        )
        start_position = end_position

    ax4.minorticks_on()
    ax4.tick_params(direction="in", which="both", top=True, right=True)
    ax4.grid(which="both", linestyle="--", linewidth=0.5)
    ax4.autoscale(enable=True, axis="x", tight=True)
    ax4.autoscale(enable=True, axis="y", tight=True)
    ax4.legend()
    ax4.set_xlabel("Thickness (nm)")
    ax4.set_ylabel("Top DBR Reflectivity")

    ax5 = plt.subplot(3, 2, 6)  # Row 3, Column 2

    ax5.plot(
        analyse_VCSEL_results.temperature_arr,
        analyse_VCSEL_results.wavelength_T_theory_arr * 1e9,
        label=f"{analyse_VCSEL_results.temperature_coefficent*1e9:.3f} nm/K",
    )
    ax5.plot(
        analyse_VCSEL_results.temperature_arr,
        analyse_VCSEL_results.cavity_resonance_arr * 1e9,
        linestyle="",
        marker="o",
        label="Simulation",
    )
    ax5.minorticks_on()
    ax5.tick_params(direction="in", which="both", top=True, right=True)
    ax5.grid(which="both", linestyle="--", linewidth=0.5)
    ax5.autoscale(enable=True, axis="x", tight=True)
    ax5.autoscale(enable=True, axis="y", tight=True)
    ax5.legend()
    ax5.set_xlabel("Active region temperature (K)")
    ax5.set_ylabel("$\\lambda_{cavity}$ (nm)")

    plt.tight_layout()

    if Save_to != None:
        formatted_time = time.strftime("%Y%m%d_%H%M%S")
        filename = f"VCSEL_analysis_{formatted_time}." + Save_to
        plt.savefig(filename, bbox_inches="tight", dpi=600)
        print("File saved to: " + filename)

    plt.show()
