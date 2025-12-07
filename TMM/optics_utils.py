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
    return np.array([[np.exp(-1j * phi), 0.0], [0.0, np.exp(1j * phi)]], dtype=complex)


def transfer_matrix(structure, wavelength):
    """

    Compute the total transfer matrix for a stacked structure. Scheme described in Coldren Example 3.1

    """
    M_total = np.eye(2, dtype=complex)

    for j in range(len(structure)):
        n_j = structure.iloc[j]["n"]
        d_j = structure.iloc[j]["d"]

        # Layer propagation
        M_layer = transfer_matrix_layer(n_j, d_j, wavelength)
        M_total = M_total @ M_layer

        # Interface to next layer (if any)
        if (j + 1) < len(structure):
            n_next = structure.iloc[j + 1]["n"]
            M_int = transfer_matrix_interface(n_j, n_next)
            M_total = M_total @ M_int

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
    T = n_transmission.real / n_incident.real * np.abs(t_field) ** 2
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
