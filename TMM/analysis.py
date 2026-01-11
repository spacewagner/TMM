# --- file: analysis.py ---
"""

TODO

High-level analysis routines for VCSELs and DBRs.

analyse_AR_coating: apply coating layer on top of structure and evaluate optical properties
analyse_etching: apply coating layer on top of structure and evaluate optical properties
analyse_lifetime_tuning: combine analyse_AR_coating and analyse_etching

analyse_cavity_dip: evaluates resonant wavelength and resonator quality as well as gain
calculate_temperature_shift: show how cavity wavelength shifts for a given temperature coefficient

analyse_VCSELs_DBRs: extracts DBRs from VCSEL structure and evaluates their optical properties
analyse_VCSEL: full analysis of VCSEL structure with the methods above, this should be the main method to use
analyse_VCSEL_lifetime_tuning: analyse Top DBR reflectivity tuning by etch and coating, as well as influence on mirrorloss and photon lifetime

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
import time
from scipy import constants as const
from dataclasses import dataclass
from typing import Tuple, List, Optional
from scipy.signal import find_peaks, peak_widths, peak_prominences

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
    flip_structure,VCSEL_embedding_active_region
)

from TMM.outputs import (
    plot_analyse_VCSEL_lifetime_tuning,
    print_analyse_VCSEL_lifetime_tuning,
    plot_analyse_VCSEL,
)


def analyse_electrical_field(
    structure, target_wavelength, position_resolution=100, Print=True, Plot=True
):
    """
    Docstring for analyse_electrical_field

    :param structure: Description
    :param target_wavelength: Description
    :param Print: Description
    :param Plot: Description

    TODO
    Confinement should just be calculated over the active region, which is a segment of the cavity

    """

    structure_field_properties_results = calculate_electrical_field(
        structure, target_wavelength, position_resolution=position_resolution, Plot=Plot
    )
    integral_full_imaginary = np.sum(
        np.imag(structure_field_properties_results.n_field_arr)
        * np.abs(structure_field_properties_results.field_values_arr) ** 2
    )
    alpha_i = coextinction_to_loss(integral_full_imaginary)
    integral_full_real = np.sum(
        # np.real(structure_field_properties_results.n_field_arr)*
        np.abs(structure_field_properties_results.field_values_arr)
        ** 2
    )

    idx_cavity = structure.loc[structure["name"] == "Cavity"].index
    idx_embedding = structure.loc[structure["name"] == "Embedding"].index
    idx_active_region = structure[structure["name"].str.contains("Active_Region")].index

    # if not idx_cavity.empty:
    #     print("Cavity without active region found")
    # if not idx_embedding.empty:
    #     print("Cavity with active region found")

    Gamma_z = 0.0
    cavity_start = None
    cavity_stop = None
    active_region_start = None
    active_region_stop = None

    if not idx_cavity.empty:
        cavity_start = structure.iloc[idx_cavity[0]]["position"]
        cavity_d = structure.iloc[idx_cavity[0]]["d"]
        cavity_stop = cavity_start + cavity_d

    if not idx_embedding.empty:
        cavity_start = structure.iloc[idx_embedding[0]]["position"]
        cavity_stop = (
            structure.iloc[idx_embedding[1]]["position"]
            + structure.iloc[idx_embedding[1]]["d"]
        )

        active_region_start = structure.iloc[idx_active_region[0]]["position"]
        active_region_stop = (
            structure.iloc[idx_active_region[-1]]["position"]
            + structure.iloc[idx_active_region[-1]]["d"]
        )

    if cavity_start != None and cavity_stop != None:
        mask_cavity = (
            structure_field_properties_results.field_positions_arr >= cavity_start
        ) & (structure_field_properties_results.field_positions_arr <= cavity_stop)
        integral_cavity = np.sum(
            np.abs(structure_field_properties_results.field_values_arr[mask_cavity])
            ** 2
        )
        Gamma_z = (
            float(integral_cavity / integral_full_real)
            if integral_full_real != 0
            else 0.0
        )

    Gamma_z_active_region = 0.0

    if active_region_start != None and active_region_stop != None:
        mask_active_region = (
            structure_field_properties_results.field_positions_arr
            >= active_region_start
        ) & (
            structure_field_properties_results.field_positions_arr <= active_region_stop
        )
        integral_active_region = np.sum(
            np.abs(
                structure_field_properties_results.field_values_arr[mask_active_region]
            )
            ** 2
        )
        Gamma_z_active_region = (
            float(integral_active_region / integral_cavity)
            if integral_cavity != 0
            else 0.0
        )

    structure_field_properties_results.alpha_i = alpha_i
    structure_field_properties_results.Gamma_z = Gamma_z
    structure_field_properties_results.Gamma_z_active_region = Gamma_z_active_region

    if Print:
        print("=" * 60 + "\nElectrical Field Analysis \n" + "=" * 60)
        print(f"Mode confinement Gamma_z: {Gamma_z:.4f}")
        print(
            f"Active region confinement Gamma_z_active_region: {Gamma_z_active_region:.4f}"
        )
        print(f"Internal loss: {alpha_i*1e-2:.5f} /cm")

    return structure_field_properties_results


@dataclass
class TuningProperties:
    d_arr: np.ndarray
    R_arr: np.ndarray
    R_tuning_range: float

    n_coating: Optional[float] = None


def analyse_AR_coating(
    structure, target_wavelength, n_coating=1.45, resolution=1e-9, Plot=True
):

    d_coating_arr = np.arange(0, target_wavelength / (2 * n_coating), resolution)
    R_arr = []

    for d_coating in d_coating_arr:
        structure_AR = apply_AR_coating(structure, n_coating, d_coating)
        M = transfer_matrix(structure_AR, target_wavelength)
        R_arr.append(calculate_reflectivity(M))

    R_tuning_range = max(R_arr) - min(R_arr)
    R_arr = np.array(R_arr)
    structure_coating_properties = TuningProperties(
        d_coating_arr, R_arr, R_tuning_range, n_coating
    )

    if Plot:
        plt.plot(d_coating_arr * 1e9, R_arr)
        plt.xlabel("Deposition Thickness (nm)")
        plt.ylabel("Reflectivity")

    return structure_coating_properties


def analyse_etching(structure, target_wavelength, resolution=1e-9, Plot=True):

    d_etch_max = 2 * (structure.iloc[-2]["d"] + structure.iloc[-3]["d"])
    d_etch_arr = np.arange(0, d_etch_max, resolution)

    R_arr = []
    for d_etch in d_etch_arr:
        structure_etch = apply_etch(structure, d_etch)
        M = transfer_matrix(structure_etch, target_wavelength)
        R_arr.append(calculate_reflectivity(M))

    top_layer_d = structure.iloc[-2]["d"]
    top_layer_d_idx = np.argmin(abs(d_etch_arr - top_layer_d))
    R_tuning_range = abs(R_arr[top_layer_d_idx] - R_arr[0])
    R_arr = np.array(R_arr)
    structure_etch_properties = TuningProperties(d_etch_arr, R_arr, R_tuning_range)

    if Plot:
        plt.plot(d_etch_arr * 1e9, R_arr)
        plt.xlabel("Etch Depth (nm)")
        plt.ylabel("Reflectivity")

    return structure_etch_properties


@dataclass
class CavityProperties:
    wavelength_arr: np.ndarray
    T_arr: np.ndarray
    resonance_wavelength: float
    resonance_R: float
    gain: float
    FWHM: float
    Q_factor: float
    photon_lifetime: float
    halfmax: float
    hw_start: float
    hw_stop: float


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

    cavity_properties_result = CavityProperties(
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

    if Plot:
        plt.plot(wavelength_arr_ROI * 1e9, t_arr_ROI_leveled, label="T (leveled)")
        plt.axvline(
            cavity_resonance_wavelength * 1e9,
            linestyle=":",
            label=f"$\\lambda$: {cavity_resonance_wavelength*1e9:.1f}nm",
        )
        plt.axhline(halfmax, linestyle=":")

        plt.hlines(
            halfmax,
            hw_start * 1e9,
            hw_stop * 1e9,
            color="red",
            label=f"FWHM: {FWHM*1e9:.2f}nm \n $\\tau_p$: {cavity_lifetime*1e12:.2f}ps",
        )
        plt.tick_params(axis="x", rotation=45)
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

    return cavity_properties_result


@dataclass
class TemperatureProperties:

    target_wavelength: float
    temperature_arr: np.ndarray
    temperature_coefficent: float
    wavelength_T_theory_arr: np.ndarray

    cavity_resonance_arr: np.ndarray
    wavelength_arr_arr: list
    r_arr_arr: list
    wavelength_tuning_range: float


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
            mesh_size=0.01e-9,
        )

        VCSEL_optical_properties_result = calculate_optical_properties(
            VCSEL_T, wavelength_arr, Plot=False
        )

        cavity_properties_result = analyse_cavity_dip(
            wavelength_arr,
            VCSEL_optical_properties_result.R_arr,
            VCSEL_optical_properties_result.T_arr,
            wavelength_T,
            fine_range,
            Plot=False,
            Print=False,
        )
        cavity_resonance_arr.append(cavity_properties_result.resonance_wavelength)
        r_arr_arr.append(VCSEL_optical_properties_result.R_arr)
        wavelength_arr_arr.append(wavelength_arr)

    cavity_resonance_arr = np.array(cavity_resonance_arr)
    wavelength_tuning_range = np.max(cavity_resonance_arr) - np.min(
        cavity_resonance_arr
    )

    results = TemperatureProperties(
        target_wavelength=target_wavelength,
        temperature_arr=T_arr,
        temperature_coefficent=temperature_coefficent,
        wavelength_T_theory_arr=wavelength_T_theory_arr,
        cavity_resonance_arr=cavity_resonance_arr,
        wavelength_arr_arr=wavelength_arr_arr,
        r_arr_arr=r_arr_arr,
        wavelength_tuning_range=wavelength_tuning_range,
    )

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

    return results


def analyse_reflectivity_tuning(
    structure, target_wavelength, n_coating=1.45, resolution=1e-9, Plot=True, Print=True
):

    structure_coating_properties = analyse_AR_coating(
        structure,
        target_wavelength,
        n_coating=n_coating,
        resolution=resolution,
        Plot=False,
    )

    structure_etching_properties = analyse_etching(
        structure, target_wavelength, resolution=resolution, Plot=False
    )

    if Plot:

        DBR_d1 = structure.iloc[-2]["d"]
        DBR_d2 = structure.iloc[-3]["d"]
        DBR_n1 = structure.iloc[-2]["n"]
        DBR_n2 = structure.iloc[-3]["n"]

        plt.plot(
            -1 * structure_etching_properties.d_arr * 1e9,
            structure_etching_properties.R_arr,
        )
        plt.plot(
            structure_coating_properties.d_arr * 1e9,
            structure_coating_properties.R_arr,
            color="tab:blue",
        )

        plt.axvline(0, color="tab:red")

        plt.axvspan(
            structure_coating_properties.d_arr[0] * 1e9,
            structure_coating_properties.d_arr[-1] * 1e9,
            alpha=0.2,
            color="tab:green",
            label="AR Coating",
        )

        start_position = structure_etching_properties.d_arr[0]
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
        print(
            f"R tuning range by AR coating: {structure_coating_properties.R_tuning_range:.4f}"
        )
        print(
            f"R tuning range by etching: {structure_etching_properties.R_tuning_range:.4f}"
        )

    return structure_coating_properties, structure_etching_properties


@dataclass
class DBR_Properties:
    DBR: pd.DataFrame
    target_wavelength: float
    wavelength_arr: np.ndarray
    R_arr: np.ndarray
    T_arr: np.ndarray
    phase_arr: np.ndarray
    idx_at_target: int
    stopband_width: float
    field_positions_arr: np.ndarray
    field_values_arr: np.ndarray
    n_field_arr: np.ndarray
    l_eff: float
    interface_position: float


def analyse_DBR(
    DBR, target_wavelength, wavelength_arr, alpha_i=5e2, Plot=True, Print=True
):
    """
    Docstring for analyse_DBR

    :param DBR: Description
    :param target_wavelength: Description
    :param wavelength_arr: Description
    :param Plot: Description
    :param Print: Description


    TODO

    Calculate penetration depth michalzik eq. 2.5
    """

    DBR_optical_properties_result = calculate_optical_properties(
        DBR, wavelength_arr, Plot=Plot
    )

    DBR_field_properties_results = calculate_electrical_field(
        DBR, target_wavelength, Plot=False
    )

    idx_target = np.argmin(np.abs(wavelength_arr - target_wavelength))
    DBR_r_at_target = DBR_optical_properties_result.R_arr[idx_target]
    DBR_phase_at_target = DBR_optical_properties_result.phase_arr[idx_target]

    idx_left = np.where(DBR_optical_properties_result.R_arr >= DBR_r_at_target * 0.99)[
        0
    ][0]
    idx_right = np.where(DBR_optical_properties_result.R_arr >= DBR_r_at_target * 0.99)[
        0
    ][-1]
    DBR_stopband_width = wavelength_arr[idx_right] - wavelength_arr[idx_left]

    n1 = DBR.loc[(DBR["name"] == "DBR_1")]["n"].values[0]
    n2 = DBR.loc[(DBR["name"] == "DBR_2")]["n"].values[0]
    l_eff = DBR_penetration_depth(n1, n2, DBR_r_at_target, target_wavelength)
    interface_position = DBR.iloc[-1]["position"]

    results = DBR_Properties(
        DBR=DBR,
        target_wavelength=target_wavelength,
        wavelength_arr=wavelength_arr,
        R_arr=DBR_optical_properties_result.R_arr,
        T_arr=DBR_optical_properties_result.T_arr,
        phase_arr=DBR_optical_properties_result.phase_arr,
        idx_at_target=int(idx_target),
        stopband_width=DBR_stopband_width,
        field_positions_arr=DBR_field_properties_results.field_positions_arr,
        field_values_arr=DBR_field_properties_results.field_values_arr,
        n_field_arr=DBR_field_properties_results.n_field_arr,
        l_eff=l_eff,
        interface_position=interface_position,
    )

    if Plot:
        plot_structure(DBR)
        plt.plot(
            np.array(DBR_field_properties_results.field_positions_arr) * 1e6,
            abs(DBR_field_properties_results.field_values_arr) ** 2
            / np.max(abs(DBR_field_properties_results.field_values_arr) ** 2)
            * np.max(DBR_field_properties_results.n_field_arr),
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
        plt.plot(wavelength_arr * 1e9, DBR_optical_properties_result.R_arr)
        plt.plot(wavelength_arr * 1e9, DBR_optical_properties_result.T_arr)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Reflectivity")
        plt.show()
        plt.plot(wavelength_arr * 1e9, DBR_optical_properties_result.phase_arr)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Phase")
        plt.legend()
        plt.show()

    if Print:

        print("=" * 60 + "\nDBR Analysis \n" + "=" * 60)
        print(f"Results at target wavelength: {target_wavelength*1e9:.3f} nm")
        print(f"DBR reflectivity: {DBR_r_at_target:.6f}")
        print(f"DBR phase: {DBR_phase_at_target:.2f}")
        print(f"DBR stopband width: {DBR_stopband_width*1e9:.2f} nm")
        print(f"DBR effective penetration depth: {l_eff*1e6:.3f} um")

    return results


@dataclass
class VCSELs_DBR_Properties:

    DBR_top: pd.DataFrame
    DBR_bottom: pd.DataFrame

    target_wavelength: float
    wavelength_arr: np.ndarray

    R_top_arr: np.ndarray
    T_top_arr: np.ndarray
    phase_top_arr: np.ndarray
    idx_at_target_top: int
    stopband_width_top: float
    field_positions_top_arr: np.ndarray
    field_values_top_arr: np.ndarray
    n_field_top_arr: np.ndarray
    l_eff_top: float
    interface_position_top: float

    R_bottom_arr: np.ndarray
    T_bottom_arr: np.ndarray
    phase_bottom_arr: np.ndarray
    idx_at_target_bottom: int
    stopband_width_bottom: float
    field_positions_bottom_arr: np.ndarray
    field_values_bottom_arr: np.ndarray
    n_field_bottom_arr: np.ndarray
    l_eff_bottom: float
    interface_position_bottom: float

    L_eff: float
    alpha_m: float
    v_gr: float
    photon_lifetime: float


def analyse_VCSELs_DBRs(
    VCSEL,
    target_wavelength,
    wavelength_arr,
    alpha_i=5e2,
    Plot=True,
    Print=True,
    Save_to="pdf",
):
    """
    Docstring for analyse_VCSELs_DBRs

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

    DBR_Bottom_properties = analyse_DBR(
        DBR_bottom, target_wavelength, wavelength_arr, Plot=False, Print=False
    )

    DBR_Top_properties = analyse_DBR(
        DBR_top, target_wavelength, wavelength_arr, Plot=False, Print=False
    )

    n_cavity = VCSEL.loc[(VCSEL["name"] == "Cavity")]["n"].values[0]
    L_cavity = VCSEL.loc[(VCSEL["name"] == "Cavity")]["d"].values[0]
    L_eff = L_cavity + DBR_Bottom_properties.l_eff + DBR_Top_properties.l_eff
    alpha_m = VCSEL_mirror_loss(
        L_eff,
        DBR_Top_properties.R_arr[DBR_Top_properties.idx_at_target],
        DBR_Bottom_properties.R_arr[DBR_Bottom_properties.idx_at_target],
    )

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
        + DBR_bottom_n_eff * DBR_Bottom_properties.l_eff
        + DBR_top_n_eff * DBR_Top_properties.l_eff
    ) / L_eff

    v_gr = const.c / n_cavity_eff

    photon_lifetime = VCSEL_photon_lifetime(v_gr, alpha_m, alpha_i)

    DBR_top_total_length = DBR_top.iloc[-1]["position"] + DBR_top.iloc[-1]["d"]

    results = VCSELs_DBR_Properties(
        DBR_top=DBR_top,
        DBR_bottom=DBR_bottom,
        target_wavelength=target_wavelength,
        wavelength_arr=wavelength_arr,
        R_top_arr=DBR_Top_properties.R_arr,
        T_top_arr=DBR_Top_properties.T_arr,
        phase_top_arr=DBR_Top_properties.phase_arr,
        idx_at_target_top=DBR_Top_properties.idx_at_target,
        stopband_width_top=DBR_Top_properties.stopband_width,
        field_positions_top_arr=DBR_Top_properties.field_positions_arr,
        field_values_top_arr=DBR_Top_properties.field_values_arr,
        n_field_top_arr=DBR_Top_properties.n_field_arr,
        l_eff_top=DBR_Top_properties.l_eff,
        interface_position_top=DBR_Top_properties.interface_position,
        R_bottom_arr=DBR_Bottom_properties.R_arr,
        T_bottom_arr=DBR_Bottom_properties.T_arr,
        phase_bottom_arr=DBR_Bottom_properties.phase_arr,
        idx_at_target_bottom=DBR_Bottom_properties.idx_at_target,
        stopband_width_bottom=DBR_Bottom_properties.stopband_width,
        field_positions_bottom_arr=DBR_Bottom_properties.field_positions_arr,
        field_values_bottom_arr=DBR_Bottom_properties.field_values_arr,
        n_field_bottom_arr=DBR_Bottom_properties.n_field_arr,
        l_eff_bottom=DBR_Bottom_properties.l_eff,
        interface_position_bottom=DBR_Bottom_properties.interface_position,
        L_eff=L_eff,
        alpha_m=alpha_m,
        v_gr=v_gr,
        photon_lifetime=photon_lifetime,
    )

    if Plot:

        # Create figure
        fig = plt.figure(figsize=(11, 8))

        ax1 = plt.subplot(2, 2, 1)  # Span column 1 of row 1

        ax1.plot(
            DBR_Bottom_properties.field_positions_arr * 1e6,
            np.real(DBR_Bottom_properties.n_field_arr),
            label="n(z)",
        )
        ax1.plot(
            DBR_Bottom_properties.field_positions_arr * 1e6,
            abs(DBR_Bottom_properties.field_values_arr) ** 2
            / np.max(abs(DBR_Bottom_properties.field_values_arr) ** 2)
            * np.max(DBR_Bottom_properties.n_field_arr),
            color="tab:red",
            label="$norm. |E|^2$",
        )
        ax1.axvline(
            (DBR_Bottom_properties.interface_position - DBR_Bottom_properties.l_eff)
            * 1e6,
            linestyle=":",
            color="black",
            label=f"l_eff = {DBR_Bottom_properties.l_eff*1e6:.2f}$\\mu m$ ",
        )
        ax1.minorticks_on()
        ax1.grid(which="both", linestyle="--", linewidth=0.5)
        ax1.tick_params(direction="in", which="both", top=True, right=True)
        ax1.autoscale(enable=True, axis="x", tight=True)
        ax1.autoscale(enable=True, axis="y", tight=True)
        ax1.set_xlabel("Position (um)")
        ax1.legend()
        ax1.set_title("Bottom DBR")

        ax2 = plt.subplot(2, 2, 2)

        ax2.plot(
            DBR_Top_properties.field_positions_arr * 1e6,
            np.flip(np.real(DBR_Top_properties.n_field_arr)),
            label="n(z)",
        )
        ax2.plot(
            DBR_Top_properties.field_positions_arr * 1e6,
            np.flip(
                abs(DBR_Top_properties.field_values_arr) ** 2
                / np.max(abs(DBR_Top_properties.field_values_arr) ** 2)
                * np.max(DBR_Top_properties.n_field_arr)
            ),
            color="tab:red",
            label="$norm. |E|^2$",
        )
        ax2.axvline(
            (
                DBR_top_total_length
                - DBR_Top_properties.interface_position
                + DBR_Top_properties.l_eff
            )
            * 1e6,
            linestyle=":",
            color="black",
            label=f"l_eff = {DBR_Top_properties.l_eff*1e6:.2f}$\\mu m$ ",
        )
        ax2.minorticks_on()
        ax2.grid(which="both", linestyle="--", linewidth=0.5)
        ax2.tick_params(direction="in", which="both", top=True, right=True)
        ax2.autoscale(enable=True, axis="x", tight=True)
        ax2.autoscale(enable=True, axis="y", tight=True)
        ax2.set_xlabel("Position (um)")
        ax2.legend()
        ax2.set_title("Top DBR")

        ax3 = plt.subplot(2, 2, 3)
        ax3.plot(
            wavelength_arr * 1e9,
            DBR_Bottom_properties.R_arr,
            label=f"$DBR_B$ {DBR_Bottom_properties.R_arr[DBR_Bottom_properties.idx_at_target]:.5}",
        )
        ax3.plot(
            wavelength_arr * 1e9,
            DBR_Top_properties.R_arr,
            label=f"$DBR_T$: {DBR_Top_properties.R_arr[DBR_Top_properties.idx_at_target]:.5}",
        )
        ax3.minorticks_on()
        ax3.grid(which="both", linestyle="--", linewidth=0.5)
        ax3.tick_params(direction="in", which="both", top=True, right=True)
        ax3.autoscale(enable=True, axis="x", tight=True)
        ax3.set_ylim(0, 1)
        ax3.set_xlabel("Wavelength (nm)")
        ax3.set_ylabel("Reflectivity")
        ax3.legend()

        ax4 = plt.subplot(2, 2, 4)
        ax4.plot(
            wavelength_arr * 1e9, DBR_Bottom_properties.phase_arr, label="Bottom DBR"
        )
        ax4.plot(wavelength_arr * 1e9, DBR_Top_properties.phase_arr, label="Top DBR")
        ax4.minorticks_on()
        ax4.grid(which="both", linestyle="--", linewidth=0.5)
        ax4.tick_params(direction="in", which="both", top=True, right=True)
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
        print(
            f"Bottom DBR reflectivity: {DBR_Bottom_properties.R_arr[DBR_Bottom_properties.idx_at_target]:.6f}"
        )
        print(
            f"Top DBR reflectivity: {DBR_Top_properties.R_arr[DBR_Top_properties.idx_at_target]:.6f}"
        )
        print(
            f"Bottom DBR phase: {DBR_Bottom_properties.phase_arr[DBR_Bottom_properties.idx_at_target]:.2f}"
        )
        print(
            f"Top DBR phase: {DBR_Top_properties.phase_arr[DBR_Top_properties.idx_at_target]:.2f}"
        )
        print(
            f"Bottom DBR stopband width: {DBR_Bottom_properties.stopband_width*1e9:.2f} nm"
        )
        print(f"Top DBR stopband width: {DBR_Top_properties.stopband_width*1e9:.2f} nm")
        print(
            f"Bottom DBR effective penetration depth: {DBR_Bottom_properties.l_eff*1e6:.3f} um"
        )
        print(
            f"Top DBR effective penetration depth: {DBR_Top_properties.l_eff*1e6:.3f} um"
        )
        print(f"Effective cavity length: {L_eff*1e6:.3f} um")
        print(f"Mirror loss: {alpha_m*1e-2:.2f} /cm")
        print(f"Internal loss (estimated): {alpha_i*1e-2:.2f} /cm")
        print(f"Group velocity in effective cavity: {v_gr:.2f} m/s")
        print(f"Estimated photon lifetime: {photon_lifetime*1e12:.2f} ps")

    return results


@dataclass
class VCSEL_Properties:

    # general
    VCSEL: pd.DataFrame
    wavelength_arr: np.ndarray
    target_wavelength: float

    # VCSEL optical
    VCSEL_R_arr: np.ndarray
    VCSEL_T_arr: np.ndarray
    VCSEL_phase_arr: np.ndarray

    # VCSEL field
    VCSEL_field_positions_arr: np.ndarray
    VCSEL_field_values_arr: np.ndarray
    VCSEL_n_field_arr: np.ndarray

    # Cavity analysis
    cavity_wavelength_arr: np.ndarray
    cavity_T_arr: np.ndarray
    cavity_resonance_wavelength: float
    cavity_resonance_R: float
    cavity_gain: float
    cavity_FWHM: float
    cavity_Q_factor: float
    cavity_photon_lifetime: float
    cavity_halfmax: float
    cavity_hw_start: float
    cavity_hw_stop: float

    # DBR analysis
    R_top_arr: np.ndarray
    T_top_arr: np.ndarray
    phase_top_arr: np.ndarray
    idx_at_target_top: int
    stopband_width_top: float
    field_positions_top_arr: np.ndarray
    field_values_top_arr: np.ndarray
    n_field_top_arr: np.ndarray
    l_eff_top: float
    interface_position_top: float

    R_bottom_arr: np.ndarray
    T_bottom_arr: np.ndarray
    phase_bottom_arr: np.ndarray
    idx_at_target_bottom: int
    stopband_width_bottom: float
    field_positions_bottom_arr: np.ndarray
    field_values_bottom_arr: np.ndarray
    n_field_bottom_arr: np.ndarray
    l_eff_bottom: float
    interface_position_bottom: float

    L_eff: float
    alpha_m: float
    v_gr: float
    photon_lifetime: float

    # Reflectivity Tuning
    d_coating_arr: np.ndarray
    R_coating_arr: np.ndarray
    R_tuning_range_coating: float
    n_coating: float

    d_etching_arr: np.ndarray
    R_etching_arr: np.ndarray
    R_tuning_range_etching: float

    # Temperature Tuning
    temperature_arr: np.ndarray
    temperature_coefficent: float
    wavelength_T_theory_arr: np.ndarray

    cavity_resonance_arr: np.ndarray
    wavelength_arr_arr: list
    r_arr_arr: list
    wavelength_tuning_range: float


def analyse_VCSEL(
    VCSEL,
    target_wavelength,
    n_coating=1.45,
    fine_range=1e-9,
    mesh_size=0.001e-9,
    plot_details=False,
    print_details=True,
    Save_to="pdf",
):

    if plot_details:
        plot_structure(VCSEL)

    wavelength_arr = wavelength_arr_adaptive_mesh(
        target_wavelength - 100e-9,
        target_wavelength + 100e-9,
        target_wavelength,
        fine_range=fine_range,
        mesh_size=mesh_size,
    )

    optical_properties_result = calculate_optical_properties(
        VCSEL, wavelength_arr, Plot=plot_details
    )
    cavity_properties_result = analyse_cavity_dip(
        wavelength_arr,
        optical_properties_result.R_arr,
        optical_properties_result.T_arr,
        target_wavelength,
        fine_range=fine_range,
        Plot=plot_details,
        Print=print_details,
    )

    VCSELs_DBR_properties = analyse_VCSELs_DBRs(
        VCSEL, target_wavelength, wavelength_arr, Plot=False
    )

    # VCSEL_coating_properties, VCSEL_etching_properties = analyse_reflectivity_tuning(
    #     VCSEL,
    #     target_wavelength,
    #     n_coating=n_coating,
    #     Plot=plot_details,
    #     Print=print_details,
    # )

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

    DBR_bottom = build_DBR_structure(
        n_bottom_1, n_bottom_2, N_bottom, target_wavelength, n_substrate, n_cavity
    )

    DBR_top = build_DBR_structure(
        n_top_1, n_top_2, N_top, target_wavelength, n_cavity, n_air
    )

    # here the fact is used that for lossles materials the transfermatrix is invariant for incident direction
    DBR_top_coating_properties = analyse_AR_coating(
        DBR_top,
        target_wavelength,
        n_coating=n_coating,
        resolution=1e-9,
        Plot=False,
    )
    plt.show()
    DBR_top_etching_properties = analyse_etching(
        DBR_top, target_wavelength, resolution=1e-9, Plot=False
    )

    VCSEL_field_properties_results = analyse_electrical_field(
        VCSEL, target_wavelength, Plot=plot_details, Print=print_details
    )

    threshold_gain = VCSEL_threshold_gain(
        VCSEL_field_properties_results.Gamma_z, VCSELs_DBR_properties.alpha_m
    )
    print(f"Estimated threshold gain: {threshold_gain*1e-2:.2f}/cm")

    T_arr = np.linspace(300, 400, 5)

    VCSEL_temperature_properties = calculate_temperature_shift(
        VCSEL,
        target_wavelength,
        T_arr,
        temperature_coefficent=0.061e-9,
        Plot=plot_details,
        Print=print_details,
    )

    results = VCSEL_Properties(
        VCSEL=VCSEL,
        wavelength_arr=wavelength_arr,
        target_wavelength=target_wavelength,
        VCSEL_R_arr=optical_properties_result.R_arr,
        VCSEL_T_arr=optical_properties_result.T_arr,
        VCSEL_phase_arr=optical_properties_result.phase_arr,
        VCSEL_field_positions_arr=VCSEL_field_properties_results.field_positions_arr,
        VCSEL_field_values_arr=VCSEL_field_properties_results.field_values_arr,
        VCSEL_n_field_arr=VCSEL_field_properties_results.n_field_arr,
        cavity_T_arr=cavity_properties_result.T_arr,
        cavity_wavelength_arr=cavity_properties_result.wavelength_arr,
        cavity_resonance_wavelength=cavity_properties_result.resonance_wavelength,
        cavity_resonance_R=cavity_properties_result.resonance_R,
        cavity_gain=cavity_properties_result.gain,
        cavity_FWHM=cavity_properties_result.FWHM,
        cavity_Q_factor=cavity_properties_result.Q_factor,
        cavity_photon_lifetime=cavity_properties_result.photon_lifetime,
        cavity_halfmax=cavity_properties_result.halfmax,
        cavity_hw_start=cavity_properties_result.hw_start,
        cavity_hw_stop=cavity_properties_result.hw_stop,
        R_top_arr=VCSELs_DBR_properties.R_top_arr,
        T_top_arr=VCSELs_DBR_properties.T_top_arr,
        phase_top_arr=VCSELs_DBR_properties.phase_top_arr,
        idx_at_target_top=VCSELs_DBR_properties.idx_at_target_top,
        stopband_width_top=VCSELs_DBR_properties.stopband_width_top,
        field_positions_top_arr=VCSELs_DBR_properties.field_positions_top_arr,
        field_values_top_arr=VCSELs_DBR_properties.field_values_top_arr,
        n_field_top_arr=VCSELs_DBR_properties.n_field_top_arr,
        l_eff_top=VCSELs_DBR_properties.l_eff_top,
        interface_position_top=VCSELs_DBR_properties.interface_position_top,
        R_bottom_arr=VCSELs_DBR_properties.R_bottom_arr,
        T_bottom_arr=VCSELs_DBR_properties.T_bottom_arr,
        phase_bottom_arr=VCSELs_DBR_properties.phase_bottom_arr,
        idx_at_target_bottom=VCSELs_DBR_properties.idx_at_target_bottom,
        stopband_width_bottom=VCSELs_DBR_properties.stopband_width_bottom,
        field_positions_bottom_arr=VCSELs_DBR_properties.field_positions_bottom_arr,
        field_values_bottom_arr=VCSELs_DBR_properties.field_values_bottom_arr,
        n_field_bottom_arr=VCSELs_DBR_properties.n_field_bottom_arr,
        l_eff_bottom=VCSELs_DBR_properties.l_eff_bottom,
        interface_position_bottom=VCSELs_DBR_properties.interface_position_bottom,
        L_eff=VCSELs_DBR_properties.L_eff,
        alpha_m=VCSELs_DBR_properties.alpha_m,
        v_gr=VCSELs_DBR_properties.v_gr,
        photon_lifetime=VCSELs_DBR_properties.photon_lifetime,
        d_coating_arr=DBR_top_coating_properties.d_arr,
        R_coating_arr=DBR_top_coating_properties.R_arr,
        R_tuning_range_coating=DBR_top_coating_properties.R_tuning_range,
        n_coating=n_coating,
        d_etching_arr=DBR_top_etching_properties.d_arr,
        R_etching_arr=DBR_top_etching_properties.R_arr,
        R_tuning_range_etching=DBR_top_etching_properties.R_tuning_range,
        temperature_arr=T_arr,
        temperature_coefficent=VCSEL_temperature_properties.temperature_coefficent,
        wavelength_T_theory_arr=VCSEL_temperature_properties.wavelength_T_theory_arr,
        cavity_resonance_arr=VCSEL_temperature_properties.cavity_resonance_arr,
        wavelength_arr_arr=VCSEL_temperature_properties.wavelength_arr_arr,
        r_arr_arr=VCSEL_temperature_properties.r_arr_arr,
        wavelength_tuning_range=VCSEL_temperature_properties.wavelength_tuning_range,
    )

    # plotting

    plot_analyse_VCSEL(results, Save_to="pdf")

    return results


@dataclass
class CavityTuningProperties:

    structure: pd.DataFrame

    alpha_i: float

    d_coating_arr: np.ndarray
    R_coating_arr: np.ndarray
    R_tuning_range_coating: float
    n_coating: float
    L_eff_coating_arr: np.ndarray
    n_cavity_eff_coating_arr: np.ndarray
    v_gr_coating_arr: np.ndarray
    alpha_m_coating_arr: np.ndarray
    alpha_m_coating_range: float
    photon_lifetime_coating_arr: np.ndarray
    photon_lifetime_coating_range: float

    d_etching_arr: np.ndarray
    R_etching_arr: np.ndarray
    R_tuning_range_etching: float
    L_eff_etch_arr: np.ndarray
    n_cavity_eff_etch_arr: np.ndarray
    v_gr_etch_arr: np.ndarray
    alpha_m_etch_arr: np.ndarray
    alpha_m_etch_range: float
    photon_lifetime_etch_arr: np.ndarray
    photon_lifetime_etch_range: float


def analyse_VCSEL_lifetime_tuning(
    VCSEL,
    target_wavelength,
    n_coating=1.45,
    alpha_i=5e2,
    resolution=1e-9,
    Plot=True,
    Print=True,
    Save_to="pdf",
):
    """
    Docstring for analyse_VCSEL_lifetime_tuning

    :param VCSEL: Description
    :param target_wavelength: Description
    :param n_coating: Description
    :param resolution: Description
    :param Plot: Description
    :param Print: Description

    Basically the analyse_reflectivity() function, but for every etch and depostionen thickness, the effective cavity length and the effective refractive index of the cavity are calculated, to calculate mirror losses and photon lifetime.

    TODO

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

    DBR_bottom = build_DBR_structure(
        n_bottom_1, n_bottom_2, N_bottom, target_wavelength, n_substrate, n_cavity
    )

    DBR_top = build_DBR_structure(
        n_top_1, n_top_2, N_top, target_wavelength, n_cavity, n_air
    )

    # here the fact is used that for lossles materials the transfermatrix is invariant for incident direction
    DBR_top_coating_properties = analyse_AR_coating(
        DBR_top,
        target_wavelength,
        n_coating=n_coating,
        resolution=resolution,
        Plot=False,
    )
    plt.show()
    DBR_top_etching_properties = analyse_etching(
        DBR_top, target_wavelength, resolution=resolution, Plot=False
    )
    plt.show()

    DBR_bottom_properties = analyse_DBR(
        DBR_bottom,
        target_wavelength,
        np.array([target_wavelength]),
        Plot=False,
        Print=False,
    )

    # this is the precise method, maybe change in effective penetration depth is neglectable, therefore DBR_top_l_eff before coating might be sufficient. Variation for ar coating < 1nm, for etch < 5nm

    n1 = DBR_top.loc[(DBR_top["name"] == "DBR_1")]["n"].values[0]
    n2 = DBR_top.loc[(DBR_top["name"] == "DBR_2")]["n"].values[0]

    DBR_top_l_eff_coating_arr = []
    for d_coating in DBR_top_coating_properties.d_arr:
        DBR_top_AR = apply_AR_coating(DBR_top, n_coating, d_coating)

        M_top_DBR = transfer_matrix(DBR_top_AR, target_wavelength)
        R_top_DBR = calculate_reflectivity(M_top_DBR)
        l_eff = DBR_penetration_depth(n1, n2, R_top_DBR, target_wavelength)
        DBR_top_l_eff_coating_arr.append(l_eff)

    DBR_top_l_eff_etch_arr = []
    for d_etch in DBR_top_etching_properties.d_arr:
        DBR_top_etch = apply_etch(DBR_top, d_etch)

        M_top_DBR = transfer_matrix(DBR_top_etch, target_wavelength)
        R_top_DBR = calculate_reflectivity(M_top_DBR)
        l_eff = DBR_penetration_depth(n1, n2, R_top_DBR, target_wavelength)
        DBR_top_l_eff_etch_arr.append(l_eff)

    n_cavity = VCSEL.loc[(VCSEL["name"] == "Cavity")]["n"].values[0]
    L_cavity = VCSEL.loc[(VCSEL["name"] == "Cavity")]["d"].values[0]

    L_cavity_etch_arr = [L_cavity] * len(DBR_top_etching_properties.d_arr)
    DBR_bottom_l_eff_etch_arr = [DBR_bottom_properties.l_eff] * len(
        DBR_top_etching_properties.d_arr
    )
    L_eff_etch_arr = (
        np.array(L_cavity_etch_arr)
        + np.array(DBR_top_l_eff_etch_arr)
        + np.array(DBR_bottom_l_eff_etch_arr)
    )

    L_cavity_coating_arr = [L_cavity] * len(DBR_top_coating_properties.d_arr)
    DBR_bottom_l_eff_coating_arr = [DBR_bottom_properties.l_eff] * len(
        DBR_top_coating_properties.d_arr
    )
    L_eff_coating_arr = (
        np.array(L_cavity_coating_arr)
        + np.array(DBR_top_l_eff_coating_arr)
        + np.array(DBR_bottom_l_eff_coating_arr)
    )

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

    alpha_m_etch_arr = []
    n_cavity_eff_etch_arr = []
    v_gr_etch_arr = []
    photon_lifetime_etch_arr = []
    for i in range(len(DBR_top_etching_properties.d_arr)):

        alpha_m_etch = VCSEL_mirror_loss(
            L_eff_etch_arr[i],
            DBR_top_etching_properties.R_arr[i],
            DBR_bottom_properties.R_arr[DBR_bottom_properties.idx_at_target],
        )
        alpha_m_etch_arr.append(alpha_m_etch)

        n_cavity_eff = (
            n_cavity * L_cavity
            + DBR_bottom_n_eff * DBR_bottom_properties.l_eff
            + DBR_top_n_eff * DBR_top_l_eff_etch_arr[i]
        ) / L_eff_etch_arr[i]

        n_cavity_eff_etch_arr.append(n_cavity_eff)

        v_gr = const.c / n_cavity_eff
        v_gr_etch_arr.append(v_gr)

        photon_lifetime_etch = VCSEL_photon_lifetime(v_gr, alpha_m_etch, alpha_i)
        photon_lifetime_etch_arr.append(photon_lifetime_etch)

    alpha_m_etch_range = np.max(alpha_m_etch_arr) - np.min(alpha_m_etch_arr)
    photon_lifetime_etch_range = np.max(photon_lifetime_etch_arr) - np.min(
        photon_lifetime_etch_arr
    )

    alpha_m_coating_arr = []
    n_cavity_eff_coating_arr = []
    v_gr_coating_arr = []
    photon_lifetime_coating_arr = []
    for i in range(len(DBR_top_coating_properties.d_arr)):

        alpha_m_coating = VCSEL_mirror_loss(
            L_eff_coating_arr[i],
            DBR_top_coating_properties.R_arr[i],
            DBR_bottom_properties.R_arr[DBR_bottom_properties.idx_at_target],
        )
        alpha_m_coating_arr.append(alpha_m_coating)

        n_cavity_eff = (
            n_cavity * L_cavity
            + DBR_bottom_n_eff * DBR_bottom_properties.l_eff
            + DBR_top_n_eff * DBR_top_l_eff_coating_arr[i]
        ) / L_eff_coating_arr[i]

        n_cavity_eff_coating_arr.append(n_cavity_eff)

        v_gr = const.c / n_cavity_eff
        v_gr_coating_arr.append(v_gr)

        photon_lifetime_coating = VCSEL_photon_lifetime(v_gr, alpha_m_coating, alpha_i)
        photon_lifetime_coating_arr.append(photon_lifetime_coating)

    alpha_m_coating_range = np.max(alpha_m_coating_arr) - np.min(alpha_m_coating_arr)
    photon_lifetime_coating_range = np.max(photon_lifetime_coating_arr) - np.min(
        photon_lifetime_coating_arr
    )

    L_eff_coating_arr = np.array(L_eff_coating_arr)
    n_cavity_eff_coating_arr = np.array(n_cavity_eff_coating_arr)
    v_gr_coating_arr = np.array(v_gr_coating_arr)
    alpha_m_coating_arr = np.array(alpha_m_coating_arr)
    photon_lifetime_coating_arr = np.array(photon_lifetime_coating_arr)

    L_eff_etch_arr = np.array(L_eff_etch_arr)
    n_cavity_eff_etch_arr = np.array(n_cavity_eff_etch_arr)
    v_gr_etch_arr = np.array(v_gr_etch_arr)
    alpha_m_etch_arr = np.array(alpha_m_etch_arr)
    photon_lifetime_etch_arr = np.array(photon_lifetime_etch_arr)

    VCSEL_cavity_tuning_properties = CavityTuningProperties(
        structure=VCSEL,
        alpha_i=alpha_i,
        d_coating_arr=DBR_top_coating_properties.d_arr,
        R_coating_arr=DBR_top_coating_properties.R_arr,
        R_tuning_range_coating=DBR_top_coating_properties.R_tuning_range,
        n_coating=n_coating,
        L_eff_coating_arr=L_eff_coating_arr,
        n_cavity_eff_coating_arr=n_cavity_eff_coating_arr,
        v_gr_coating_arr=v_gr_coating_arr,
        alpha_m_coating_arr=alpha_m_coating_arr,
        alpha_m_coating_range=alpha_m_coating_range,
        photon_lifetime_coating_arr=photon_lifetime_coating_arr,
        photon_lifetime_coating_range=photon_lifetime_coating_range,
        d_etching_arr=DBR_top_etching_properties.d_arr,
        R_etching_arr=DBR_top_etching_properties.R_arr,
        R_tuning_range_etching=DBR_top_etching_properties.R_tuning_range,
        L_eff_etch_arr=L_eff_etch_arr,
        n_cavity_eff_etch_arr=n_cavity_eff_etch_arr,
        v_gr_etch_arr=v_gr_etch_arr,
        alpha_m_etch_arr=alpha_m_etch_arr,
        alpha_m_etch_range=alpha_m_etch_range,
        photon_lifetime_etch_arr=photon_lifetime_etch_arr,
        photon_lifetime_etch_range=photon_lifetime_etch_range,
    )

    if Plot:
        plot_analyse_VCSEL_lifetime_tuning(
            VCSEL_cavity_tuning_properties, Save_to=Save_to
        )

    if Print:
        print_analyse_VCSEL_lifetime_tuning(VCSEL_cavity_tuning_properties)

    return VCSEL_cavity_tuning_properties


@dataclass
class ActiveRegionEmbeddingProperties:
    d_embedding_arr: np.ndarray
    Gamma_z_arr: np.ndarray
    Gamma_z_active_region_arr: np.ndarray

    d_optimimum_arr: np.ndarray
    Gamma_z_optimimum_arr: np.ndarray

    d_active_region: float

def optimize_embedding_thickness(VCSEL, active_region, target_wavelength, d_min, d_max, d_resolution=20):

    d_embedding_arr = np.linspace(d_min, d_max, d_resolution)
    Gamma_z_arr = []
    Gamma_z_active_region_arr = []
    position_resolution = 100

    d_active_region = active_region["d"].sum()

    for d_embedding in d_embedding_arr:
        VCSEL_modified = VCSEL_embedding_active_region(
            VCSEL, active_region, d_embedding=d_embedding
        )
        idx_embedding = VCSEL_modified.loc[VCSEL_modified["name"] == "Embedding"].index
        results = analyse_electrical_field(
            VCSEL_modified,
            target_wavelength,
            position_resolution=position_resolution,
            Plot=False,
            Print=False,
        )
        plt.plot(
            results.field_positions_arr[
                idx_embedding[0]
                * position_resolution : (idx_embedding[1] + 1)
                * position_resolution
            ],
            np.abs(
                results.field_values_arr[
                    idx_embedding[0]
                    * position_resolution : (idx_embedding[1] + 1)
                    * position_resolution
                ]
            )
            ** 2
            / np.max(np.abs(results.field_values_arr) ** 2),
        )
        Gamma_z = results.Gamma_z
        Gamma_z_arr.append(Gamma_z)

        Gamma_z_active_region = results.Gamma_z_active_region
        Gamma_z_active_region_arr.append(Gamma_z_active_region)

    plt.show()

    Gamma_z_arr = np.array(Gamma_z_arr)
    Gamma_z_active_region_arr = np.array(Gamma_z_active_region_arr)

    Gamma_z_optimimum_arr = []
    Gamma_z_optimimum_positions_arr = []

    peaks, properties = find_peaks(
        Gamma_z_arr,
        height=0,
        threshold=None,
        distance=1,
    )

    Gamma_z_optimimum_arr = Gamma_z_arr[peaks]
    d_optimimum_arr = d_embedding_arr[peaks]

    results = ActiveRegionEmbeddingProperties(
        d_embedding_arr=d_embedding_arr,
        Gamma_z_arr=Gamma_z_arr,
        Gamma_z_active_region_arr=Gamma_z_active_region_arr,
        d_optimimum_arr=d_optimimum_arr,
        Gamma_z_optimimum_arr=Gamma_z_optimimum_arr,
        d_active_region=d_active_region,
    )

    plt.plot(
        (results.d_embedding_arr + results.d_active_region) * 1e9,
        results.Gamma_z_arr,
        label="Γ Cavity",
    )
    plt.plot(
        (results.d_embedding_arr + results.d_active_region) * 1e9,
        results.Gamma_z_active_region_arr,
        label="Γ Active Region",
    )

    for pos, val in zip(results.d_optimimum_arr, results.Gamma_z_optimimum_arr):
        plt.plot((pos + results.d_active_region) * 1e9, val, marker="o", color="red")

    plt.xlabel("Cavity Length (nm)")
    plt.ylabel("Mode Confinement Γz")
    plt.legend()

    ax2 = plt.gca().twiny()
    ax2.set_xlim(plt.xlim())
    ax2.set_xlabel("Embedding Thickness (nm)")
    ax2.tick_params(axis="x")

    cavity_lengths = plt.gca().get_xticks()
    ax2.set_xticks(cavity_lengths)
    ax2.set_xticklabels(
        np.linspace(
            results.d_embedding_arr[0] * 1e9,
            results.d_embedding_arr[-1] * 1e9,
            len(cavity_lengths),
        ).round(1)
    )

    plt.tight_layout()
    plt.show()
    return results