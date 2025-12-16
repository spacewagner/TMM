# --- file: analysis.py ---
"""

TODO

need to be adjusted:
analyze_DBR
analyze_VCSELs_DBR
analyze_VCSEL

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
from matplotlib import colormaps
import time
from scipy import constants as const

from TMM.optics_utils import (
    wavelength_to_frequency,
    transfer_matrix,
    calculate_reflectivity,
    DBR_penetration_depth,
    VCSEL_mirror_loss,
    VCSEL_photon_lifetime,
    VCSEL_threshold_gain,
    coextinction_to_loss,
)
from TMM.field_solver import calculate_optical_properties, calculate_electrical_field

from TMM.structure_builder import (
    apply_AR_coating,
    VCSEL_temperature_correction,
    cavity_wavelength_temperature_correction,
    wavelength_arr_adaptive_mesh,
    apply_etch,
    get_VCSEL_structure,
    build_DBR_structure,
    plot_structure,
    interpolate_structure,
    flip_structure,
)


def analyze_electrical_field(structure, target_wavelength, Print=True, Plot=True):

    field_positions, field_values, n_field_arr = calculate_electrical_field(
        structure, target_wavelength, Plot=Plot
    )

    field_positions = np.array(field_positions)
    n_field_arr = np.array(n_field_arr)

    integral_full_real = np.sum(np.real(n_field_arr) * np.abs(field_values) ** 2)
    integral_full_imaginary = np.sum(np.imag(n_field_arr) * np.abs(field_values) ** 2)
    alpha_i = coextinction_to_loss(integral_full_imaginary)

    Gamma_z = 0.0
    if (structure["name"] == "Cavity").any():
        cavity_start = float(
            structure.loc[structure["name"] == "Cavity", "position"].iloc[0]
        )
        cavity_d = float(structure.loc[structure["name"] == "Cavity", "d"].iloc[0])
        cavity_stop = cavity_start + cavity_d

        mask_cavity = (field_positions >= cavity_start) & (
            field_positions <= cavity_stop
        )
        integral_cavity = np.sum(
            np.real(n_field_arr[mask_cavity]) * np.abs(field_values[mask_cavity]) ** 2
        )
        Gamma_z = (
            float(integral_cavity / integral_full_real)
            if integral_full_real != 0
            else 0.0
        )

    if Print:
        print("=" * 60 + "\nElectrical Field Analysis \n" + "=" * 60)
        print(f"Mode confinement Gamma_z: {Gamma_z:.4f}")
        print(f"Internal loss: {alpha_i*1e-2:.5f} /cm")

    return field_positions, field_values, n_field_arr, Gamma_z, alpha_i


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
    if cavity_gain > 1e-9:
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
        print(f"Photon lifetime: {cavity_lifetime*1e12:.2f} ps")

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
    fine_range=5e-9,
):

    cavity_resonance_arr = []
    r_arr_arr = []

    wavelength_arr_arr = []

    wavelength_T_theory_arr = cavity_wavelength_temperature_correction(
        target_wavelength, T_arr, temperature_coefficent
    )

    for T in T_arr:

        VCSEL_T, wavelength_T = VCSEL_temperature_correction(
            VCSEL, target_wavelength, T, temperature_coefficent
        )

        wavelength_arr = wavelength_arr_adaptive_mesh(
            wavelength_T - 100e-9,
            wavelength_T + 100e-9,
            wavelength_T,
            fine_range,
            mesh_size=0.1e-9,
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

        plt.autoscale(enable=True, axis="x", tight=True)
        plt.autoscale(enable=True, axis="y", tight=True)
        plt.xlabel("Thickness (nm)")
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


def analyze_DBR(DBR, target_wavelength, wavelength_arr, Plot=True, Print=True):
    """
    Docstring for analyze_DBR

    :param DBR: Description
    :param target_wavelength: Description
    :param wavelength_arr: Description
    :param Plot: Description
    :param Print: Description


    TODO

    Calculate penetration depth michalzik eq. 2.5
    """

    wavelength_arr, DBR_r_arr, DBR_t_arr, DBR_phase_arr = calculate_optical_properties(
        DBR, wavelength_arr, Plot=Plot
    )

    DBR_field_positions, DBR_field_values, DBR_n_field_arr = calculate_electrical_field(
        DBR, target_wavelength, Plot=False
    )

    idx_target = np.argmin(np.abs(wavelength_arr - target_wavelength))
    DBR_r_at_target = DBR_r_arr[idx_target]
    DBR_phase_at_target = DBR_phase_arr[idx_target]

    idx_left = np.where(DBR_r_arr >= DBR_r_at_target * 0.99)[0][0]
    idx_right = np.where(DBR_r_arr >= DBR_r_at_target * 0.99)[0][-1]
    DBR_stopband_width = wavelength_arr[idx_right] - wavelength_arr[idx_left]

    n1 = DBR.loc[(DBR["name"] == "DBR_1")]["n"].values[0]
    n2 = DBR.loc[(DBR["name"] == "DBR_2")]["n"].values[0]
    l_eff = DBR_penetration_depth(n1, n2, DBR_r_at_target, target_wavelength)
    interface_position = DBR.iloc[-1]["position"]

    if Plot:
        plot_structure(DBR)
        plt.plot(
            np.array(DBR_field_positions) * 1e6,
            abs(DBR_field_values) ** 2
            / np.max(abs(DBR_field_values) ** 2)
            * np.max(DBR_n_field_arr),
            color="tab:red",
            label="$|E|^2$",
        )
        plt.axvline(
            (interface_position - l_eff) * 1e6,
            linestyle=":",
            color="black",
            label=f"l_eff = {l_eff*1e6:.2f}$\\mu m$ ",
        )
        plt.legend()
        plt.show()
        plt.plot(wavelength_arr * 1e9, DBR_r_arr)
        plt.plot(wavelength_arr * 1e9, DBR_t_arr)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Reflectivity")
        plt.show()
        plt.plot(wavelength_arr * 1e9, DBR_phase_arr)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Phase")
        plt.legend()
        plt.show()

    if Print:

        print("=" * 60 + "\nDBR Analysis \n" + "=" * 60)
        print(f"Results at target wavelength: {target_wavelength*1e9:.3f}nm")
        print(f"DBR reflectivity: {DBR_r_at_target:.6f}")
        print(f"DBR phase: {DBR_phase_at_target:.2f}")
        print(f"DBR stopband width: {DBR_stopband_width*1e9:.2f}nm")
        print(f"DBR effective penetration depth: {l_eff*1e6:.3f}$\\mu m$")

    return (
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
    )


def analyze_VCSELs_DBRs(
    VCSEL, target_wavelength, wavelength_arr, Plot=True, Print=True, Save_to="pdf"
):
    """
    Docstring for analyze_VCSELs_DBRs

    :param VCSEL: Description
    :param target_wavelength: Description
    :param wavelength_arr: Description
    :param Plot: Description
    :param Print: Description
    """

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

    # Top DBR needs to be flipped, since incident wave is coming from bottom side, not top side
    DBR_top = flip_structure(DBR_top)

    (
        DBR_bottom_r_arr,
        DBR_bottom_t_arr,
        DBR_bottom_phase_arr,
        DBR_bottom_r_at_target,
        DBR_bottom_phase_at_target,
        DBR_bottom_stopband_width,
        DBR_bottom_field_positions,
        DBR_bottom_field_values,
        DBR_bottom_n_field_arr,
        DBR_bottom_l_eff,
        DBR_bottom_interface_position,
    ) = analyze_DBR(
        DBR_bottom, target_wavelength, wavelength_arr, Plot=False, Print=False
    )

    (
        DBR_top_r_arr,
        DBR_top_t_arr,
        DBR_top_phase_arr,
        DBR_top_r_at_target,
        DBR_top_phase_at_target,
        DBR_top_stopband_width,
        DBR_top_field_positions,
        DBR_top_field_values,
        DBR_top_n_field_arr,
        DBR_top_l_eff,
        DBR_top_interface_position,
    ) = analyze_DBR(DBR_top, target_wavelength, wavelength_arr, Plot=False, Print=False)

    n_cavity = VCSEL.loc[(VCSEL["name"] == "Cavity")]["n"].values[0]
    L_cavity = VCSEL.loc[(VCSEL["name"] == "Cavity")]["d"].values[0]
    L_eff = L_cavity + DBR_bottom_l_eff + DBR_top_l_eff
    alpha_m = VCSEL_mirror_loss(L_eff, DBR_top_r_at_target, DBR_bottom_r_at_target)

    DBR_bottom_n1 = DBR_bottom.loc[(DBR_bottom["name"] == "DBR_1")]["n"].values[0]
    DBR_bottom_n2 = DBR_bottom.loc[(DBR_bottom["name"] == "DBR_2")]["n"].values[0]
    DBR_bottom_d1 = DBR_bottom.loc[(DBR_bottom["name"] == "DBR_1")]["d"].values[0]
    DBR_bottom_d2 = DBR_bottom.loc[(DBR_bottom["name"] == "DBR_2")]["d"].values[0]
    DBR_bottom_n_eff = (
        DBR_bottom_n1 * DBR_bottom_d1 + DBR_bottom_n2 * DBR_bottom_d2
    ) / (DBR_bottom_d1 + DBR_bottom_d2)

    DBR_top_n1 = DBR_top.loc[(DBR_top["name"] == "DBR_1")]["n"].values[0]
    DBR_top_n2 = DBR_top.loc[(DBR_top["name"] == "DBR_2")]["n"].values[0]
    DBR_top_d1 = DBR_top.loc[(DBR_top["name"] == "DBR_1")]["d"].values[0]
    DBR_top_d2 = DBR_top.loc[(DBR_top["name"] == "DBR_2")]["d"].values[0]
    DBR_top_n_eff = (DBR_top_n1 * DBR_top_d1 + DBR_top_n2 * DBR_top_d2) / (
        DBR_top_d1 + DBR_top_d2
    )

    n_cavity_eff = (
        n_cavity * L_cavity
        + DBR_bottom_n_eff * DBR_bottom_l_eff
        + DBR_top_n_eff * DBR_top_l_eff
    ) / L_eff

    v_gr = const.c / n_cavity_eff

    photon_lifetime = VCSEL_photon_lifetime(v_gr, alpha_m)

    if Plot:

        # Create figure
        fig = plt.figure(figsize=(11, 8))

        ax1 = plt.subplot(2, 2, 1)  # Span column 1 of row 1

        ax1.plot(
            np.array(DBR_bottom_field_positions) * 1e6,
            np.real(DBR_bottom_n_field_arr),
            label="n(z)",
        )
        ax1.plot(
            np.array(DBR_bottom_field_positions) * 1e6,
            abs(DBR_bottom_field_values) ** 2
            / np.max(abs(DBR_bottom_field_values) ** 2)
            * np.max(DBR_bottom_n_field_arr),
            color="tab:red",
            label="$norm. |E|^2$",
        )
        ax1.axvline(
            (DBR_bottom_interface_position - DBR_bottom_l_eff) * 1e6,
            linestyle=":",
            color="black",
            label=f"l_eff = {DBR_bottom_l_eff*1e6:.2f}$\\mu m$ ",
        )
        ax1.autoscale(enable=True, axis="x", tight=True)
        ax1.autoscale(enable=True, axis="y", tight=True)
        ax1.set_xlabel("Position (nm)")
        ax1.legend()
        ax1.set_title("Bottom DBR")

        ax2 = plt.subplot(2, 2, 2)

        ax2.plot(
            np.array(DBR_top_field_positions) * 1e6,
            np.real(DBR_top_n_field_arr),
            label="n(z)",
        )
        ax2.plot(
            np.array(DBR_top_field_positions) * 1e6,
            abs(DBR_top_field_values) ** 2
            / np.max(abs(DBR_top_field_values) ** 2)
            * np.max(DBR_top_n_field_arr),
            color="tab:red",
            label="$norm. |E|^2$",
        )
        ax2.axvline(
            (DBR_top_interface_position - DBR_top_l_eff) * 1e6,
            linestyle=":",
            color="black",
            label=f"l_eff = {DBR_top_l_eff*1e6:.2f}$\\mu m$ ",
        )
        ax2.autoscale(enable=True, axis="x", tight=True)
        ax2.autoscale(enable=True, axis="y", tight=True)
        ax2.set_xlabel("Position (nm)")
        ax2.legend()
        ax2.set_title("Top DBR")

        ax3 = plt.subplot(2, 2, 3)
        ax3.plot(wavelength_arr * 1e9, DBR_bottom_r_arr, label="Bottom DBR")
        ax3.plot(wavelength_arr * 1e9, DBR_top_r_arr, label="Top DBR")
        ax3.autoscale(enable=True, axis="x", tight=True)
        ax3.set_ylim(0, 1)
        ax3.set_xlabel("Wavelength (nm)")
        ax3.set_ylabel("Reflectivity")
        ax3.legend()

        ax4 = plt.subplot(2, 2, 4)
        ax4.plot(wavelength_arr * 1e9, DBR_bottom_phase_arr, label="Bottom DBR")
        ax4.plot(wavelength_arr * 1e9, DBR_top_phase_arr, label="Top DBR")
        ax4.autoscale(enable=True, axis="x", tight=True)
        ax4.set_ylim(-np.pi, np.pi)
        ax4.set_xlabel("Wavelength (nm)")
        ax4.set_ylabel("Phase")
        ax4.legend()

        plt.tight_layout()

        if Save_to != None:
            formatted_time = time.strftime("%Y%m%d_%H%M%S")
            filename = f"DBR_analysis_{formatted_time}." + Save_to
            plt.savefig(filename, bbox_inches="tight", dpi=600)
            print("File saved to: " + filename)

    if Print:

        print("=" * 60 + "\nDBR Analysis \n" + "=" * 60)
        print(f"Results at target wavelength: {target_wavelength*1e9:.3f} nm")
        print(f"Bottom DBR reflectivity: {DBR_bottom_r_at_target:.6f}")
        print(f"Top DBR reflectivity: {DBR_top_r_at_target:.6f}")
        print(f"Bottom DBR phase: {DBR_bottom_phase_at_target:.2f}")
        print(f"Top DBR phase: {DBR_top_phase_at_target:.2f}")
        print(f"Bottom DBR stopband width: {DBR_bottom_stopband_width*1e9:.2f} nm")
        print(f"Top DBR stopband width: {DBR_top_stopband_width*1e9:.2f} nm")
        print(f"Bottom DBR effective penetration depth: {DBR_bottom_l_eff*1e6:.3f} um")
        print(f"Top DBR effective penetration depth: {DBR_top_l_eff*1e6:.3f} um")
        print(f"Effective cavity length: {L_eff*1e6:.3f} um")
        print(f"Mirror loss: {alpha_m*1e-2:.2f} /cm")
        print(f"Group velocity in effective cavity: {v_gr:.2f} m/s")
        print(f"Estimated photon lifetime: {photon_lifetime*1e12:.2f} ps")

    return (
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
        photon_lifetime,
    )


def analyze_VCSEL(
    VCSEL,
    target_wavelength,
    n_coating=1.45,
    plot_details=False,
    print_details=True,
    Save_to="pdf",
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
        photon_lifetime,
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
        n_AR_coating=n_coating,
        Plot=plot_details,
        Print=print_details,
    )

    field_positions, field_values, n_field_arr, Gamma_z, alpha_i = (
        analyze_electrical_field(
            VCSEL, target_wavelength, Plot=plot_details, Print=print_details
        )
    )

    threshold_gain = VCSEL_threshold_gain(Gamma_z, alpha_m)
    print(f"Estimated threshold gain: {threshold_gain*1e-2:.2f}/cm")

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

    structure_interpolated = interpolate_structure(VCSEL)
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
    n_cav = float(
        structure_interpolated.loc[
            structure_interpolated["name"] == "Cavity", "n"
        ].iloc[0]
    )
    ax1.fill_between(
        np.array([cavity_start, cavity_stop]) * 1e6,
        [0, 0],
        [n_cav, n_cav],
        alpha=0.3,
        color="tab:red",
        label="Cavity",
    )
    ax1.plot(
        np.array(field_positions) * 1e6,
        abs(field_values) ** 2 / np.max(abs(field_values) ** 2) * np.max(n_field_arr),
        color="tab:red",
        label="$|E|^2$",
    )

    # Combine them
    ax1.legend()
    ax1.autoscale(enable=True, axis="x", tight=True)
    ax1.autoscale(enable=True, axis="y", tight=True)
    ax1.set_xlabel("Postion $(\\mu m)$")
    ax1.set_ylabel("Refractive index")

    # Second row: two plots
    ax2 = plt.subplot(3, 2, 3)  # Row 2, Column 1
    ax1 = plt.gca()
    ax1.plot(wavelength_arr * 1e9, r_arr, color="tab:blue", label="Reflectivity")
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Reflectivity")

    # Create second y-axis for phase
    ax2 = ax1.twinx()
    ax2.plot(
        wavelength_arr * 1e9,
        phase_arr / np.pi,
        color="tab:orange",
        label="Phase",
        linestyle=":",
    )
    ax2.set_ylabel("Phase (rad)", color="tab:orange")

    # Add a single legend for both plots
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    ax2.set_ylim(-1, 1)

    if cavity_resonance_reflectivity <= 1:
        ax1.set_ylim(0, 1)
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
