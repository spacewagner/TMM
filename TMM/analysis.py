# --- file: analysis.py ---
"""

High-level analysis routines for VCSELs and DBRs.

analyze_AR_coating: apply coating layer on top of structure and evaluate optical properties
analyze_etching: apply coating layer on top of structure and evaluate optical properties
analyze_lifetime_tuning: combine analyze_AR_coating and analyze_etching


analyse_cavity_dip: evaluates resonant wavelength and resonator quality as well as gain
calculate_temperature_shift: show how cavity wavelength shifts for a given temperature coefficient

analyze_VCSELs_DBRs: extracts DBRs from VCSEL structure and evaluates their optical properties
analyze_VCSEL: full analysis of VCSEL structure with the methods above, this should be the main method to use


"""

import numpy as np
import matplotlib.pyplot as plt
import time

from optics_utils import wavelength_to_frequency
from field_solver import calculate_optical_properties, calculate_electrical_field

from structure_builder import (
    apply_AR_coating,
    VCSEL_temperature_correction,
    cavity_wavelength_temperature_correction,
    wavelength_arr_adaptive_mesh,
    apply_etch,
    get_VCSEL_structure,
    build_DBR_structure,
    plot_structure,
)

from optics_utils import transfer_matrix, calculate_reflectivity
from matplotlib import colormaps


def analyze_AR_coating(structure, target_wavelength, n_coating=1.45, Plot=True):

    d_coating_arr = np.arange(0, target_wavelength / (2 * n_coating), 1e-9)
    R_arr = []

    for d_coating in d_coating_arr:
        structure_AR = apply_AR_coating(structure, n_coating, d_coating)
        M = transfer_matrix(structure_AR, target_wavelength)
        R_arr.append(calculate_reflectivity(M))

    if Plot:
        plt.plot(d_coating_arr * 1e9, R_arr)
        plt.xlabel("Deposition Thickness (nm)")
        plt.ylabel("Reflectivity")

    R_tuning_range = max(R_arr) - min(R_arr)

    return d_coating_arr, R_arr, R_tuning_range


def analyze_etching(structure, target_wavelength, Plot=True):

    d_etch_max = 2 * (structure.iloc[-2]["d"] + structure.iloc[-3]["d"])
    d_etch_arr = np.arange(0, d_etch_max, 1e-9)

    R_arr = []
    for d_etch in d_etch_arr:
        structure_etch = apply_etch(structure, d_etch)
        M = transfer_matrix(structure_etch, target_wavelength)
        R_arr.append(calculate_reflectivity(M))

    top_layer_d = structure.iloc[-2]["d"]
    top_layer_d_idx = np.argmin(abs(d_etch_arr - top_layer_d))
    R_tuning_range = abs(R_arr[top_layer_d_idx] - R_arr[0])

    if Plot:
        plt.plot(d_etch_arr * 1e9, R_arr)
        plt.xlabel("Etch Depth (nm)")
        plt.ylabel("Reflectivity")

    return d_etch_arr, R_arr, R_tuning_range


def analyse_cavity_dip(
    wavelength_arr,
    R_arr,
    T_arr,
    target_wavelength,
    fine_range=5e-9,
    Plot=True,
    Print=True,
):
    wavelength_min_index = np.argmin(
        abs(wavelength_arr - target_wavelength + fine_range)
    )
    wavelength_max_index = np.argmin(
        abs(wavelength_arr - target_wavelength - fine_range)
    )

    wavelength_arr_ROI = wavelength_arr[wavelength_min_index:wavelength_max_index]
    r_arr_ROI = R_arr[wavelength_min_index:wavelength_max_index]
    t_arr_ROI = T_arr[wavelength_min_index:wavelength_max_index]
    cavity_resonance_reflectivity = min(r_arr_ROI)
    cavity_resonance_transmission = max(t_arr_ROI)
    cavity_gain = cavity_resonance_reflectivity + cavity_resonance_transmission - 1
    if cavity_gain > 0:
        idx_peak = np.argmax(r_arr_ROI)
    else:
        idx_peak = np.argmin(r_arr_ROI)
    cavity_resonance_wavelength = wavelength_arr_ROI[idx_peak]

    # subtract baseline and locate peak
    t_arr_ROI_leveled = t_arr_ROI - np.min(t_arr_ROI)

    halfmax = np.max(t_arr_ROI_leveled) / 2.0

    left_idx = np.where(t_arr_ROI_leveled[0:idx_peak] <= halfmax)[0]
    right_idx = np.where(t_arr_ROI_leveled[idx_peak:] <= halfmax)[0]

    if len(left_idx) > 0:
        hw_start = wavelength_arr_ROI[left_idx[-1]]
    else:
        hw_start = wavelength_arr_ROI[0]

    if len(right_idx) > 0:
        hw_stop = wavelength_arr_ROI[idx_peak + right_idx[0]]
    else:
        hw_stop = wavelength_arr_ROI[-1]

    FWHM = hw_stop - hw_start if hw_stop > hw_start else np.nan
    Q_factor = (
        cavity_resonance_wavelength / FWHM if FWHM and not np.isnan(FWHM) else np.inf
    )

    f, omega = wavelength_to_frequency(target_wavelength, 1.0)
    cavity_lifetime = Q_factor / omega if np.isfinite(Q_factor) else np.inf

    if Plot:
        plt.plot(wavelength_arr_ROI * 1e9, t_arr_ROI_leveled, label="T (leveled)")
        plt.axvline(
            cavity_resonance_wavelength * 1e9,
            linestyle="--",
            label=f"$\\lambda$: {cavity_resonance_wavelength*1e9:.1f}nm",
        )
        plt.axhline(halfmax, linestyle="--")

        plt.hlines(
            halfmax,
            hw_start * 1e9,
            hw_stop * 1e9,
            color="red",
            label=f"FWHM: {FWHM*1e9:.2f}",
        )

        plt.xlabel("Wavelength (nm)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    if Print:
        print("=" * 60 + "\nCavity Analysis \n" + "=" * 60)
        print(f"Cavity wavelength (peak): {cavity_resonance_wavelength*1e9:.1f} nm")
        print(f"Cavity reflectivity: {cavity_resonance_reflectivity:.3f}")
        print(f"Cavity transmission: {cavity_resonance_transmission:.3f}")
        print(f"Cavity gain: {cavity_gain:.3f}")
        print(f"FWHM: {FWHM*1e9:.4f} nm -> Q: {Q_factor:.2f}")
        print(f"Cavity lifetime: {cavity_lifetime:.4e} s")

    return (
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
    )


def calculate_temperature_shift(
    VCSEL,
    target_wavelength,
    T_arr,
    temperature_coefficent=0.061e-9,
    Plot=True,
    Print=True,
):

    cavity_resonance_arr = []
    r_arr_arr = []

    fine_range = 5e-9
    wavelength_arr_arr = []

    wavelength_T_theory_arr = cavity_wavelength_temperature_correction(
        target_wavelength, T_arr, temperature_coefficent
    )

    for T in T_arr:

        VCSEL_T, wavelength_T = VCSEL_temperature_correction(
            VCSEL, target_wavelength, T, temperature_coefficent
        )

        wavelength_arr = wavelength_arr_adaptive_mesh(
            wavelength_T - 100e-9, wavelength_T + 100e-9, wavelength_T, fine_range
        )

        wavelength_arr, r_arr, t_arr, phase_arr = calculate_optical_properties(
            VCSEL_T, wavelength_arr, Plot=False
        )

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
        ) = analyse_cavity_dip(
            wavelength_arr,
            r_arr,
            t_arr,
            wavelength_T,
            fine_range,
            Plot=False,
            Print=False,
        )
        cavity_resonance_arr.append(cavity_resonance_wavelength)
        r_arr_arr.append(r_arr)
        wavelength_arr_arr.append(wavelength_arr)

    wavelength_tuning_range = max(cavity_resonance_arr) - min(cavity_resonance_arr)

    if Plot:

        plt.plot(
            T_arr,
            wavelength_T_theory_arr * 1e9,
            label=f"{temperature_coefficent*1e9:.3f} nm/K",
        )
        plt.plot(
            T_arr,
            np.array(cavity_resonance_arr) * 1e9,
            linestyle="",
            marker="o",
            label="Simulation",
        )
        plt.legend()
        plt.xlabel("Active region temperature (K)")
        plt.ylabel("Cavity resonance wavelength (nm)")
        plt.show()

        for i in range(len(r_arr_arr)):
            plt.plot(wavelength_arr_arr[i] * 1e9, r_arr_arr[i] + i)
            plt.text(
                cavity_resonance_arr[i] * 1e9,
                i + 0.8,
                str(T_arr[i]) + "K",
                color=colormaps["tab10"](i),
            )

        plt.xlabel("Wavelength~(nm)")
        plt.ylabel("Power reflectance~(unitless)")
        plt.show()

    if Print:

        print("=" * 60 + "\nTemperature Analysis \n" + "=" * 60)
        print(
            f"Wavelength tuning range over 300K-400K: {wavelength_tuning_range*1e9:.1f} nm"
        )

    return (
        T_arr,
        wavelength_T_theory_arr,
        wavelength_arr_arr,
        cavity_resonance_arr,
        temperature_coefficent,
    )


def analyze_lifetime_tuning(
    structure, target_wavelength, n_AR_coating=1.45, Plot=True, Print=True
):

    d_coating_arr, R_coating_arr, R_coating_tuning_range = analyze_AR_coating(
        structure, target_wavelength, Plot=False
    )
    d_etch_arr, R_etch_arr, R_etch_tuning_range = analyze_etching(
        structure, target_wavelength, Plot=False
    )

    if Plot:

        DBR_d1 = structure.iloc[-2]["d"]
        DBR_d2 = structure.iloc[-3]["d"]
        DBR_n1 = structure.iloc[-2]["n"]
        DBR_n2 = structure.iloc[-3]["n"]

        plt.plot(-1 * d_etch_arr * 1e9, R_etch_arr)
        plt.plot(d_coating_arr * 1e9, R_coating_arr, color="tab:blue")

        plt.axvline(0, color="tab:red")

        plt.axvspan(
            d_coating_arr[0] * 1e9,
            d_coating_arr[-1] * 1e9,
            alpha=0.2,
            color="tab:green",
            label="AR Coating",
        )

        start_position = d_etch_arr[0]
        label = None
        for i in range(2):

            end_position = start_position + DBR_d1
            if i == 1:
                label = DBR_n1
            plt.axvspan(
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
            plt.axvspan(
                -1 * start_position * 1e9,
                -1 * end_position * 1e9,
                alpha=0.2,
                color="tab:orange",
                label=label,
            )
            start_position = end_position

        plt.xlabel("Thickness~(nm)")
        plt.ylabel("Reflectivity")
        plt.legend()

    if Print:

        print("=" * 60 + "\nLifetime Tuning Analysis \n" + "=" * 60)
        print(f"R tuning range by AR coating: {R_coating_tuning_range:.4f}")
        print(f"R tuning range by etching: {R_etch_tuning_range:.4f}")

    return (
        d_coating_arr,
        R_coating_arr,
        R_coating_tuning_range,
        n_AR_coating,
        d_etch_arr,
        R_etch_arr,
        R_etch_tuning_range,
    )


def analyze_VCSELs_DBRs(
    VCSEL, target_wavelength, wavelength_arr, Plot=True, Print=True
):

    VCSEL_real = VCSEL.copy()
    VCSEL_real["n"] = np.real(VCSEL_real["n"])  # only take real part of n

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
    ) = get_VCSEL_structure(VCSEL_real)

    DBR_bottom = build_DBR_structure(
        n_bottom_1, n_bottom_2, N_bottom, target_wavelength, n_substrate, n_cavity
    )

    DBR_top = build_DBR_structure(
        n_top_1, n_top_2, N_top, target_wavelength, n_cavity, n_air
    )

    wavelength_arr, DBR_bottom_r_arr, DBR_bottom_t_arr, DBR_bottom_phase_arr = (
        calculate_optical_properties(DBR_bottom, wavelength_arr, Plot=False)
    )

    wavelength_arr, DBR_top_r_arr, DBR_top_t_arr, DBR_top_phase_arr = (
        calculate_optical_properties(DBR_top, wavelength_arr, Plot=False)
    )

    idx_target = np.argmin(np.abs(wavelength_arr - target_wavelength))
    DBR_top_r_at_target = DBR_top_r_arr[idx_target]
    DBR_bottom_r_at_target = DBR_bottom_r_arr[idx_target]
    DBR_top_phase_at_target = DBR_top_phase_arr[idx_target]
    DBR_bottom_phase_at_target = DBR_bottom_phase_arr[idx_target]

    idx_left_bottom = np.where(DBR_bottom_r_arr >= DBR_bottom_r_at_target * 0.99)[0][0]
    idx_right_bottom = np.where(DBR_bottom_r_arr >= DBR_bottom_r_at_target * 0.99)[0][
        -1
    ]
    DBR_bottom_stopband_width = (
        wavelength_arr[idx_right_bottom] - wavelength_arr[idx_left_bottom]
    )

    idx_left_top = np.where(DBR_top_r_arr >= DBR_top_r_at_target * 0.99)[0][0]
    idx_right_top = np.where(DBR_top_r_arr >= DBR_top_r_at_target * 0.99)[0][-1]
    DBR_top_stopband_width = (
        wavelength_arr[idx_right_top] - wavelength_arr[idx_left_top]
    )

    if Plot:
        plot_structure(DBR_top)
        plt.title("Bottom DBR")
        plot_structure(DBR_bottom)
        plt.title("Top DBR")
        plt.show()
        plt.plot(wavelength_arr * 1e9, DBR_bottom_r_arr, label="Bottom DBR")
        plt.plot(wavelength_arr * 1e9, DBR_top_r_arr, label="Top DBR")
        plt.xlabel("Wavelength~(nm)")
        plt.ylabel("Reflectivity")
        plt.legend()
        plt.show()
        plt.plot(wavelength_arr * 1e9, DBR_bottom_phase_arr, label="Bottom DBR")
        plt.plot(wavelength_arr * 1e9, DBR_top_phase_arr, label="Top DBR")
        plt.xlabel("Wavelength~(nm)")
        plt.ylabel("Phase")
        plt.legend()
        plt.show()

    if Print:

        print("=" * 60 + "\nDBR Analysis \n" + "=" * 60)
        print(f"Results at target wavelength: {target_wavelength*1e9:.3f} nm")
        print(f"Bottom DBR reflectivity: {DBR_bottom_r_at_target:.6f}")
        print(f"Top DBR reflectivity: {DBR_top_r_at_target:.6f}")
        print(f"Bottom DBR phase: {DBR_bottom_phase_at_target:.2f}")
        print(f"Top DBR phase: {DBR_top_phase_at_target:.2f}")
        print(f"Bottom DBR stopband width: {DBR_bottom_stopband_width*1e9:.2f}")
        print(f"Top DBR stopband width: {DBR_top_stopband_width*1e9:.2f}")

    return (
        DBR_bottom_r_arr,
        DBR_bottom_t_arr,
        DBR_bottom_phase_arr,
        DBR_top_r_arr,
        DBR_top_t_arr,
        DBR_top_phase_arr,
    )


def analyze_VCSEL(
    VCSEL, target_wavelength, plot_details=False, print_details=True, Save_to="pdf"
):

    if plot_details:
        plot_structure(VCSEL)

    wavelength_arr = wavelength_arr_adaptive_mesh(
        target_wavelength - 100e-9, target_wavelength + 100e-9, target_wavelength, 5e-9
    )

    wavelength_arr, r_arr, t_arr, phase_arr = calculate_optical_properties(
        VCSEL, wavelength_arr, Plot=plot_details
    )
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
    ) = analyse_cavity_dip(
        wavelength_arr,
        r_arr,
        t_arr,
        target_wavelength,
        Plot=plot_details,
        Print=print_details,
    )

    (
        DBR_bottom_r_arr,
        DBR_bottom_t_arr,
        DBR_bottom_phase_arr,
        DBR_top_r_arr,
        DBR_top_t_arr,
        DBR_top_phase_arr,
    ) = analyze_VCSELs_DBRs(VCSEL, target_wavelength, wavelength_arr, Plot=False)

    (
        d_coating_arr,
        R_coating_arr,
        R_coating_tuning_range,
        n_AR_coating,
        d_etch_arr,
        R_etch_arr,
        R_etch_tuning_range,
    ) = analyze_lifetime_tuning(
        VCSEL,
        target_wavelength,
        n_AR_coating=1.45,
        Plot=plot_details,
        Print=print_details,
    )

    field_positions, field_values, Gamma_z = calculate_electrical_field(
        VCSEL, target_wavelength, Plot=plot_details, Print=print_details
    )
    T_arr = np.linspace(300, 400, 5)

    (
        T_arr,
        wavelength_T_theory_arr,
        wavelength_arr_arr,
        cavity_resonance_arr,
        temperature_coefficent,
    ) = calculate_temperature_shift(
        VCSEL,
        target_wavelength,
        T_arr,
        temperature_coefficent=0.061e-9,
        Plot=plot_details,
        Print=print_details,
    )

    # plotting

    # Create figure
    fig = plt.figure(figsize=(11, 8))

    # First row: full width plot (row 1, spans 2 columns)
    ax1 = plt.subplot(3, 2, (1, 2))  # Span columns 1-2 of row 1

    # only plot real part of n
    VCSEL_real = VCSEL.copy()
    VCSEL_real["n"] = np.real(VCSEL_real["n"])

    ax1.step(
        VCSEL_real["position"] * 1e6,
        VCSEL_real["n"],
        where="post",
        #  label="n(z)"
    )

    last_pos = VCSEL_real.iloc[-1]["position"]
    last_d = VCSEL_real.iloc[-1]["d"]
    ax1.step(
        np.array([last_pos, last_pos + last_d]) * 1e6,
        [VCSEL_real.iloc[-1]["n"], VCSEL_real.iloc[-1]["n"]],
        where="post",
        color="tab:blue",
    )

    cavity_start = float(VCSEL_real.loc[VCSEL["name"] == "Cavity", "position"].iloc[0])
    cavity_d = float(VCSEL_real.loc[VCSEL["name"] == "Cavity", "d"].iloc[0])
    cavity_stop = cavity_start + cavity_d
    n_cav = float(VCSEL_real.loc[VCSEL["name"] == "Cavity", "n"].iloc[0])
    ax1.fill_between(
        np.array([cavity_start, cavity_stop]) * 1e6,
        [0, 0],
        [n_cav, n_cav],
        alpha=0.3,
        color="tab:red",
        label="Cavity",
    )

    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.plot(
        field_positions * 1e6,
        np.abs(field_values) ** 2,
        # label="$|E|^2$",
        color="tab:red",
    )

    # Get lines and labels from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    # Combine them
    ax1.legend(lines1 + lines2, labels1 + labels2)
    ax1.autoscale(enable=True, axis="x", tight=True)
    ax1.autoscale(enable=True, axis="y", tight=True)
    ax2.autoscale(enable=True, axis="y", tight=True)

    ax1.set_xlabel("Postion $(\\mu m)$")
    ax1.set_ylabel("Refractive index", color="tab:blue")
    ax2.set_ylabel("$|E|^2$", color="tab:red")

    # Second row: two plots
    ax2 = plt.subplot(3, 2, 3)  # Row 2, Column 1
    ax1 = plt.gca()
    ax1.plot(wavelength_arr * 1e9, r_arr, color="tab:blue", label="Reflectivity")
    ax1.axhline(0.5, linestyle=":", color="tab:red")
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Reflectivity", color="tab:blue")

    # Create second y-axis for phase
    ax2 = ax1.twinx()
    ax2.plot(
        wavelength_arr * 1e9,
        phase_arr / np.pi,
        color="tab:orange",
        label="Phase",
        linestyle="--",
    )
    ax2.set_ylabel("Phase (rad)", color="tab:orange")

    # # Add a single legend for both plots
    # lines1, labels1 = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    if cavity_resonance_reflectivity <= 1:
        ax1.set_ylim(0, 1)
    ax2.set_ylim(-1, 1)
    ax1.autoscale(enable=True, axis="x", tight=True)

    ax3 = plt.subplot(3, 2, 4)  # Row 2, Column 2
    ax3.plot(wavelength_arr_ROI * 1e9, t_arr_ROI_leveled, label="T (leveled)")
    ax3.axvline(
        cavity_resonance_wavelength * 1e9,
        linestyle="--",
        label=f"$\\lambda$: {cavity_resonance_wavelength*1e9:.1f}nm",
    )
    ax3.axhline(halfmax, linestyle="--")

    ax3.hlines(
        halfmax,
        hw_start * 1e9,
        hw_stop * 1e9,
        color="red",
        label=f"FWHM: {FWHM*1e9:.2f}nm",
    )

    ax3.set_xlabel("Wavelength (nm)")
    ax3.legend()
    ax3.autoscale(enable=True, axis="x", tight=True)
    ax3.autoscale(enable=True, axis="y", tight=True)

    # Third row: two plots
    ax4 = plt.subplot(3, 2, 5)  # Row 3, Column 1

    DBR_d1 = VCSEL.iloc[-2]["d"]
    DBR_d2 = VCSEL.iloc[-3]["d"]
    DBR_n1 = VCSEL.iloc[-2]["n"]
    DBR_n2 = VCSEL.iloc[-3]["n"]

    ax4.plot(-1 * d_etch_arr * 1e9, R_etch_arr)
    ax4.plot(d_coating_arr * 1e9, R_coating_arr, color="tab:blue")

    ax4.axvline(0, color="tab:red")

    ax4.axvspan(
        d_coating_arr[0] * 1e9,
        d_coating_arr[-1] * 1e9,
        alpha=0.2,
        color="tab:green",
        label=n_AR_coating,
    )

    start_position = d_etch_arr[0]
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

    ax4.autoscale(enable=True, axis="x", tight=True)
    ax4.autoscale(enable=True, axis="y", tight=True)
    ax4.legend()
    ax4.set_xlabel("Thickness (nm)")
    ax4.set_ylabel("Reflectivity")

    ax5 = plt.subplot(3, 2, 6)  # Row 3, Column 2

    ax5.plot(
        T_arr,
        wavelength_T_theory_arr * 1e9,
        label=f"{temperature_coefficent*1e9:.3f} nm/K",
    )
    ax5.plot(
        T_arr,
        np.array(cavity_resonance_arr) * 1e9,
        linestyle="",
        marker="o",
        label="Simulation",
    )
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
