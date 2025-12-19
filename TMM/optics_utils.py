# --- file: optics_utils.py ---
"""

This module contains low-level functions for building transfer matrices
for interfaces and layers, and for computing reflectivity, transmission,
and wavelength-frequency conversions.

Helpful literature:
Coldren: Laserdiodes and Photonic Integrated Circiuts, Chapter 3 "Mirrors and Resonators for Diode Lasers"
Chuang: Physics of Photonic Devices, Chapter 5.7 - 5.9 "Matrix Optics"


"""

import numpy as np
from scipy import constants as const


def r_fresnel(n1, n2):
    return (n1 - n2) / (n1 + n2)


def t_fresnel(n1, n2):
    return 2 * n1 / (n1 + n2)


def transfer_matrix_interface(n1: complex, n2: complex) -> np.ndarray:
    """Return the 2x2 transfer matrix for a normal-incidence interface.

    Uses the standard 2x2 approach where fields are [forward, backward].

    Transfer from n1 to n2

    Coldren Tab. 3.3

    """
    r12 = r_fresnel(n1, n2)
    t12 = t_fresnel(n1, n2)
    return np.array([[1 / t12, r12 / t12], [r12 / t12, 1 / t12]], dtype=complex)


def transfer_matrix_layer(n: complex, d: float, wavelength: float) -> np.ndarray:
    """Return the 2x2 transfer matrix for a homogeneous layer.

    The layer matrix describes the phase accumulated by forward/backward
    travelling plane waves: diag(exp(-i k d), exp(+i k d)).

    Adaption from Coldren Tab. 3.3

    """
    k = 2 * np.pi * n / wavelength
    phi = k * d
    # the notation -1j/+1j sets +- sign in extinction coefficient. Should be diag(-, +)
    return np.array([[np.exp(-1j * phi), 0.0], [0.0, np.exp(1j * phi)]], dtype=complex)


# def transfer_matrix(structure, wavelength):
#     """

#     Compute the total transfer matrix for a stacked structure. Scheme described in Coldren Example 3.1

#     """
#     M_total = np.eye(2, dtype=complex)

#     # go backwards through structure
#     for i in range(len(structure) - 1, -1, -1):

#         n = structure.iloc[i]["n"]
#         d = structure.iloc[i]["d"]

#         # Layer propagation
#         P = transfer_matrix_layer(n, d, wavelength)
#         M_total = P @ M_total

#         # Interface to previous layer, from end to start position
#         if i > 0:
#             n_previous = structure.iloc[i - 1]["n"]
#             T_n_previous_n = transfer_matrix_interface(n_previous, n)
#             M_total = T_n_previous_n @ M_total

#     return M_total


# This way it will calculate from Surface to interface
def transfer_matrix(structure, wavelength):
    """

    Compute the total transfer matrix for a stacked structure. Scheme described in Coldren Example 3.1

    """
    M_total = np.eye(2, dtype=complex)

    # go backwards through structure
    for i in range(len(structure)):

        n = structure.iloc[i]["n"]
        d = structure.iloc[i]["d"]

        # Layer propagation
        P = transfer_matrix_layer(n, d, wavelength)
        M_total = P @ M_total

        # Interface from previous layer, from end to start position
        if (i + 1) < len(structure):
            n_previous = structure.iloc[i + 1]["n"]
            T_n_previous_n = transfer_matrix_interface(n_previous, n)
            M_total = T_n_previous_n @ M_total

    return M_total


def calculate_reflectivity(M: np.ndarray) -> float:
    """Compute power reflectivity from the total transfer matrix

    Chuang eq. 5.9.37

    """
    r = M[1, 0] / M[0, 0]
    return np.abs(r) ** 2


def calculate_transmission(M: np.ndarray, n_incident, n_transmission) -> float:
    """Compute power transmission through the stack

    Chuang eq. 5.9.38 + Power-normalisation due to impedance-change of medium

    """
    t_field = 1.0 / M[0, 0]
    impedance_factor = n_transmission.real / n_incident.real
    T = impedance_factor * np.abs(t_field) ** 2
    return T


def calculate_phase(M: np.ndarray) -> float:
    """Return the phase (radians) of the reflected field (arg of r).

    Chuang eq. 5.9.37

    """
    r = M[1, 0] / M[0, 0]
    return np.angle(r)


def wavelength_to_frequency(wavelength_vacuum: float, n_medium=1.0):
    """Convert vacuum wavelength to frequency and angular frequency in a medium."""
    wavelength_medium = wavelength_vacuum / n_medium
    f = const.c / wavelength_medium
    omega = 2 * np.pi * f
    return f, omega


def R_theoretical(N, n1, n2, ns, n0):
    """

    Larisch eq. 2.3 and 2.4
    Chuang eq. 5.9.52
    Chuang eq. 11.2.2

    """

    if N == int(N):
        a = ns * (n2 / n1) ** (2 * N) - n0
        b = ns * (n2 / n1) ** (2 * N) + n0

    else:
        a = n1**2 * (n1 / n2) ** (2 * int(N)) - n0 * ns
        b = n1**2 * (n1 / n2) ** (2 * int(N)) + n0 * ns
    return (a / b) ** 2


def DBR_stopband_width():
    """
    TODO
    Michalzik eq. 2.3
    """


def DBR_penetration_depth(n1, n2, R, target_wavelength):
    """
    Michalzik eq. 2.5
    """
    delta_n = abs(n1 - n2)
    kappa = 2 * delta_n / target_wavelength
    a = np.sqrt(R)
    b = 2 * kappa
    return a / b


def VCSEL_mirror_loss(L_eff, R_top, R_bottom):
    """
    Docstring for VCSEL_mirror_loss

    :param L_eff: Description
    :param R_top: Lossless Top DBR
    :param R_bottom: Lossless Bottom DBR

    Michalzik eq. 2.17
    """
    a = 1 / L_eff
    b = np.log(1 / np.sqrt(R_top * R_bottom))
    alpha_m = a * b
    return alpha_m


def VCSEL_photon_lifetime(v_gr, alpha_m, alpha_i):
    """
    Docstring for VCSEL_threshold_gain

    :param v_gr: Description
    :param alpha_i: Internal loss, estimated
    :param alpha_m: Description

    Michalzik eq. 2.17
    """

    return 1 / (v_gr * (alpha_i + alpha_m))


def VCSEL_threshold_gain(Gamma, alpha_m, alpha_i=5e2):
    return (alpha_i + alpha_m) / Gamma


def coextinction_to_loss(kappa):
    return (4 * np.pi / const.c) * kappa
