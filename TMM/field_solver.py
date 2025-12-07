# --- file: field_solver.py ---
"""

This module enables computation of optical spectra and internal field distributions for a stack

Helpful literature:
Coldren: Laserdiodes and Photonic Integrated Circiuts, Chapter 3 "Mirrors and Resonators for Diode Lasers"
Chuang: Physics of Photonic Devices, Chapter 5.7 - 5.9 "Matrix Optics"

"""

import numpy as np
import matplotlib.pyplot as plt

from optics_utils import (
    transfer_matrix,
    calculate_reflectivity,
    calculate_transmission,
    calculate_phase,
    transfer_matrix_layer,
    transfer_matrix_interface,
)

from structure_builder import plot_structure


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
    structure, target_wavelength, position_resolution: int = 1000, Plot=True, Print=True
):
    """

    Take incident field (Amplitude_forward = 1, Amplitude_backward = 0)
    Then follow the incident field while propagating through the structure, with transfer between interfaces and propagation through layers. Interpolation through layers allows space resolution of field.


    """

    structure = structure.copy().reset_index(drop=True)
    structure["position"] = structure["position"] - structure["position"].min()

    total_length = structure.iloc[-1]["position"] + structure.iloc[-1]["d"]
    x_positions = np.linspace(0.0, total_length, position_resolution)

    E_forward = 1.0 + 0j
    E_backward = 0.0 + 0j

    field_positions = []
    field_values = []
    n_arr = []

    for i in structure.index:
        n = structure.loc[i, "n"]
        layer_start = structure.loc[i, "position"]
        d = structure.loc[i, "d"]
        layer_end = layer_start + d

        if i == 0:
            E_f = E_forward
            E_b = E_backward
        else:
            n_prev = structure.loc[i - 1, "n"]

            M_int = transfer_matrix_interface(n_prev, n)

            vec = M_int @ np.array([E_forward, E_backward], dtype=complex)
            E_f = vec[0]
            E_b = vec[1]

        in_layer = (x_positions >= layer_start) & (x_positions <= layer_end)
        x_layer = x_positions[in_layer] - layer_start

        for x_local in x_layer:
            field_positions.append(layer_start + x_local)

            M_local = transfer_matrix_layer(n, x_local, target_wavelength)
            vec_local = M_local @ np.array([E_f, E_b], dtype=complex)
            E_f_local = vec_local[0]
            E_b_local = vec_local[1]
            E_total = E_f_local + E_b_local
            field_values.append(E_total)
            n_arr.append(n)

        M_layer = transfer_matrix_layer(n, d, target_wavelength)
        vec = M_layer @ np.array([E_f, E_b], dtype=complex)

        E_forward = vec[0]
        E_backward = vec[1]

    field_positions = np.array(field_positions)
    field_values = np.array(field_values)
    n_arr = np.array(n_arr)

    Gamma_z = 0.0
    if (structure["name"] == "Cavity").any():
        cavity_start = float(
            structure.loc[structure["name"] == "Cavity", "position"].iloc[0]
        )
        cavity_d = float(structure.loc[structure["name"] == "Cavity", "d"].iloc[0])
        cavity_stop = cavity_start + cavity_d

        energy_full = np.sum(np.real(n_arr) * np.abs(field_values) ** 2)
        mask_cav = (field_positions >= cavity_start) & (field_positions <= cavity_stop)
        energy_cav = np.sum(
            np.real(n_arr[mask_cav]) * np.abs(field_values[mask_cav]) ** 2
        )
        Gamma_z = float(energy_cav / energy_full) if energy_full != 0 else 0.0

    if Plot:
        plot_structure(structure)
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax2.plot(
            field_positions * 1e6,
            np.abs(field_values) ** 2,
            label="$|E|^2$",
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

    if Print:
        print("=" * 60 + "\nElectrical Field Analysis \n" + "=" * 60)
        print(f"Mode confinement Gamma_z: {Gamma_z:.4f}")

    return field_positions, field_values, Gamma_z
