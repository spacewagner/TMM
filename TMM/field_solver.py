# --- file: field_solver.py ---
"""

This module enables computation of optical spectra and internal field distributions for a stack

Helpful literature:
Coldren: Laserdiodes and Photonic Integrated Circiuts, Chapter 3 "Mirrors and Resonators for Diode Lasers"
Chuang: Physics of Photonic Devices, Chapter 5.7 - 5.9 "Matrix Optics"

"""

import numpy as np
import matplotlib.pyplot as plt

from TMM.optics_utils import (
    transfer_matrix,
    calculate_reflectivity,
    calculate_transmission,
    calculate_phase,
    transfer_matrix_layer,
    transfer_matrix_interface,
)

from TMM.structure_builder import plot_structure, interpolate_structure


def calculate_optical_properties(structure, wavelength_arr, Plot=True):
    """

    Calculate transfer matrices of the given structure for different wavelength and extract reflectivity and transmission.

    """

    r_arr = []
    t_arr = []
    phase_arr = []

    n_incident = structure.iloc[0]["n"]
    n_transmission = structure.iloc[-1]["n"]

    for wavelength in wavelength_arr:
        M_total = transfer_matrix(structure, wavelength)
        r_arr.append(calculate_reflectivity(M_total))
        t_arr.append(calculate_transmission(M_total, n_incident, n_transmission))
        phase_arr.append(calculate_phase(M_total))

    r_arr = np.array(r_arr)
    t_arr = np.array(t_arr)
    phase_arr = np.array(phase_arr)

    if Plot:
        plt.figure()
        plt.plot(wavelength_arr, r_arr, label="Reflection")
        plt.plot(wavelength_arr, t_arr, label="Transmission", alpha=0.6)
        plt.plot(wavelength_arr, r_arr + t_arr, label="R+T", alpha=0.6)
        plt.xlabel("Wavelength (m)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

        plt.figure()
        plt.plot(wavelength_arr, phase_arr)
        plt.xlabel("Wavelength (m)")
        plt.ylabel("Phase (rad)")
        plt.grid(alpha=0.3)
        plt.show()

        # Create first y-axis for reflection
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

        # Add a single legend for both plots
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

        ax1.set_ylim(0, 1)
        ax2.set_ylim(-1, 1)

        plt.title("Reflectivity and Phase vs Wavelength")

    return wavelength_arr, r_arr, t_arr, phase_arr


def calculate_electrical_field(
    structure, target_wavelength, position_resolution: int = 100, Plot=True, Print=True
):
    structure_interpolated = interpolate_structure(structure, position_resolution)

    M_total = np.eye(2, dtype=complex)

    n_transmission = structure.iloc[-1]["n"]

    Ef = 1.0 + 0.0j
    Eb = 0.0 + 0.0j
    vec = [Ef, Eb]

    vec_arr = []
    n_field_arr = []

    # n_previous referrs to n of the previous investigated layer, so the right next layer. From point of forwards propagating wave, it's always going from the current layer to the previous investigated layer T_n_n_previous
    n_previous = structure.iloc[-1]["n"]
    field_positions = []

    # go backwards through structure
    for i in range(len(structure_interpolated) - 1, -1, -1):

        n = structure_interpolated.iloc[i]["n"]
        d = structure_interpolated.iloc[i]["d"]
        position_global = structure_interpolated.iloc[i]["position"]

        if n != n_previous:
            T_n_n_previous = transfer_matrix_interface(n, n_previous)
            vec = T_n_n_previous @ vec
            n_previous = n
            d = 0  # skip propagation in case of interface

            M_total = T_n_n_previous @ M_total

        P = transfer_matrix_layer(n, d, target_wavelength)
        vec = P @ vec

        M_total = P @ M_total

        field_positions.append(position_global)
        vec_arr.append(vec)
        n_field_arr.append(n.real)

    # collect field components
    field_values = np.array([vec[0] + vec[1] for vec in vec_arr])
    field_forward = np.array([vec[0] for vec in vec_arr])
    field_backward = np.array([vec[1] for vec in vec_arr])

    if Plot:
        # calculate R, T from transfermatrix
        n_incident = structure.iloc[0]["n"]
        n_transmission = structure.iloc[-1]["n"]
        M = transfer_matrix(structure, target_wavelength)
        R = calculate_reflectivity(M)
        T = calculate_transmission(M, n_incident, n_transmission)

        # not exactly pointing, because no norming to vacuum impedance
        S_forward = np.array(n_field_arr) * abs(field_forward) ** 2
        S_backward = np.array(n_field_arr) * abs(field_backward) ** 2

        # plotting
        plot_structure(structure)
        plt.plot(
            np.array(field_positions) * 1e6,
            abs(field_values) ** 2
            / np.max(abs(field_values) ** 2)
            * np.max(n_field_arr),
            color="tab:red",
            label="$|E|^2$",
        )
        plt.legend()
        plt.show()

        # investigate forward and backward field for consistency
        # plot needs to be normed to T and n_transmission, because initial condition was vec = [1, 0] but now should be [T/n, 0]
        plt.plot(
            field_positions, S_forward * T / n_transmission.real, label="S_forward"
        )
        plt.plot(
            field_positions, S_backward * T / n_transmission.real, label="S_backward"
        )

        # check for energy conservation
        plt.plot(
            field_positions,
            (S_forward - S_backward) * T / n_transmission.real,
            label="S_forward - S_backward",
        )

        # compare with results from M
        plt.axhline(R, linestyle=":", label="R", color="tab:orange")
        plt.axhline(T, linestyle=":", label="T", color="tab:blue")
        plt.axhline(R + T, linestyle=":", label="R+T", color="black")
        plt.plot(
            field_positions,
            n_field_arr / np.max(n_field_arr),
            alpha=0.3,
        )
        plt.legend()
        plt.show()

        # phase of forward and backward wave
        # plt.plot(field_positions, np.angle(field))
        # plt.plot(field_positions, np.angle(field_forward))
        # plt.plot(field_positions, np.angle(field_backward))
        # plt.plot(structure_interpolated["position"], structure_interpolated["n"], alpha=0.3)
        # plt.show()

    return field_positions, field_values, n_field_arr
