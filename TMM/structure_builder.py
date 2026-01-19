# --- file: structure_builder.py ---
"""

Helpers to build and visualise VCSEL/DBR structures


"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def wavelength_arr_adaptive_mesh(min, max, target, fine_range=5e-9, mesh_size=0.1e-9):
    """

    min wavelength
    max wavelength
    target wavelength
    fine_range around target_wavelength with higher mesh accuracy

    """

    segment1 = np.arange(min, target - fine_range - 1e-9, 1e-9)
    segment2 = np.arange(target - fine_range, target + fine_range, mesh_size)
    segment3 = np.arange(target + fine_range, max, 1e-9)

    result = np.concatenate([segment1, segment2, segment3])
    return result


def build_DBR_structure(
    n_1, n_2, N_pairs, target_wavelength, n_substrate=3.52, n_air=1.0
):

    # Quarter-wave thicknesses
    d_1 = target_wavelength / (4 * n_1.real)
    d_2 = target_wavelength / (4 * n_2.real)

    position = 0.0

    # Substrate (incident side)
    substrate = pd.DataFrame(
        {
            "name": ["Cladding Bottom"],
            "n": [n_substrate],
            "d": [1e-6],
            "position": [position],
        }
    )
    structure = substrate.copy()
    position += 1e-6

    # Bottom DBR
    for i in range(int(N_pairs)):
        pair = pd.DataFrame(
            {
                "name": ["DBR_1", "DBR_2"],
                "n": [n_1, n_2],
                "d": [d_1, d_2],
                "position": [position, position + d_1],
            }
        )
        structure = pd.concat([structure, pair], ignore_index=True)
        position += d_1 + d_2

    if N_pairs != int(N_pairs):
        extra = pd.DataFrame(
            {
                "name": ["DBR_1"],
                "n": [n_1],
                "d": [d_1],
                "position": [position],
            }
        )
        structure = pd.concat([structure, extra], ignore_index=True)
        position += d_1

    # Air (transmission side)
    air = pd.DataFrame(
        {"name": ["Air"], "n": [n_air], "d": [1e-6], "position": [position]}
    )
    structure = pd.concat([structure, air], ignore_index=True)

    return structure


def apply_AR_coating(structure, n_coating, d_coating):

    n_air = structure.iloc[-1]["n"]
    d_air = structure.iloc[-1]["d"]
    position = structure.iloc[-1]["position"]

    structure_AR = structure.iloc[:-1]

    if d_coating != 0:
        coating = pd.DataFrame(
            {
                "name": ["Coating"],
                "n": [n_coating],
                "d": [d_coating],
                "position": [position],
            }
        )
        structure_AR = pd.concat([structure_AR, coating], ignore_index=True)

        position += d_coating

    # Air (transmission side)
    air = pd.DataFrame(
        {"name": ["Air"], "n": [n_air], "d": [d_air], "position": [position]}
    )
    structure_AR = pd.concat([structure_AR, air], ignore_index=True)

    return structure_AR


def apply_etch(structure, d_etch):
    """
    TODO
    WORK IN PROGRESS
    """

    n_air = structure.iloc[-1]["n"]
    d_air = structure.iloc[-1]["d"]
    position = structure.iloc[-1]["position"]
    structure_etch = structure.iloc[:-1]  # remove air

    d_top = structure_etch.iloc[-1]["d"]
    d_etch_second_layer = d_etch - d_top  # if > 0: second layer is etched

    if d_etch != 0:

        while d_etch_second_layer > 0:
            structure_etch = structure_etch.iloc[:-1]  # remove top
            d_etch = (
                d_etch_second_layer  # only remove this fraction from second top layer
            )
            d_top = structure_etch.iloc[-1]["d"]
            d_etch_second_layer = d_etch - d_top

        name_layer = structure_etch.iloc[-1]["name"]
        n = structure_etch.iloc[-1]["n"]
        position = structure_etch.iloc[-1]["position"]
        d = structure_etch.iloc[-1]["d"]

        structure_etch = structure_etch.iloc[:-1]  # remove top

        if d != d_etch:

            top_layer_etched = pd.DataFrame(
                {
                    "name": [name_layer + "_etched"],
                    "n": [n],
                    "d": [d - d_etch],
                    "position": [position],
                }
            )
            structure_etch = pd.concat(
                [structure_etch, top_layer_etched], ignore_index=True
            )

        position += d - d_etch

    # Air (transmission side)
    air = pd.DataFrame(
        {"name": ["Air"], "n": [n_air], "d": [d_air], "position": [position]}
    )
    structure_etch = pd.concat([structure_etch, air], ignore_index=True)

    return structure_etch


def build_VCSEL_structure(
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
    N_cavity=1,
):

    # Quarter-wave thicknesses
    d_cavity = target_wavelength / (2 * n_cavity.real) * N_cavity

    bottom_DBR = build_DBR_structure(
        n_bottom_1, n_bottom_2, N_bottom, target_wavelength, n_substrate=n_substrate
    )
    bottom_DBR = bottom_DBR[:-1]  # remove air layer

    top_DBR = build_DBR_structure(
        n_top_1, n_top_2, N_top, target_wavelength, n_air=n_air
    )
    top_DBR = top_DBR[1:]  # remove substrate layer

    cavity = pd.DataFrame(
        {"name": ["Cavity"], "n": [n_cavity], "d": [d_cavity], "position": [0]}
    )

    VCSEL = bottom_DBR.copy()
    VCSEL = pd.concat([VCSEL, cavity], ignore_index=True)
    VCSEL = pd.concat([VCSEL, top_DBR], ignore_index=True)

    for i in range(len(VCSEL)):
        if i == 0:
            VCSEL.loc[VCSEL.index[i], "position"] = 0
        else:
            VCSEL.loc[VCSEL.index[i], "position"] = (
                VCSEL.iloc[i - 1]["position"] + VCSEL.iloc[i - 1]["d"]
            )

    return VCSEL


def plot_structure(structure, show_cavity_color: bool = True):

    structure_interpolated = interpolate_structure(structure)
    structure_interpolated["n"] = np.real(
        structure_interpolated["n"]
    )  # only plot Re(n)

    plt.title("Layer structure (refractive index vs position)")

    plt.plot(
        structure_interpolated["position"] * 1e6,
        structure_interpolated["n"],
        label="n(z)",
    )

    if (structure_interpolated["name"] == "Cavity").any() and show_cavity_color:
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
        # n_cav = float(
        #     structure_interpolated.loc[
        #         structure_interpolated["name"] == "Cavity", "n"
        #     ].iloc[0]
        # )

        n_max = structure["n"].values.max()
        plt.fill_between(
            np.array([cavity_start, cavity_stop]) * 1e6,
            [0, 0],
            [n_max, n_max],
            alpha=0.3,
            color="tab:red",
            label="Cavity",
        )

    plt.xlabel("Position (um)")
    plt.ylabel("Refractive index n")
    plt.legend()


def get_VCSEL_structure(VCSEL):

    index_cavity = list(VCSEL.loc[(VCSEL["name"] == "Cavity")].index)[0]

    DBR_bottom = VCSEL.iloc[1:index_cavity]
    DBR_top = VCSEL.iloc[index_cavity + 1 : -1]

    N_bottom = len(DBR_bottom) / 2
    N_top = len(DBR_top) / 2

    n_bottom_1 = DBR_bottom.iloc[0]["n"]
    n_bottom_2 = DBR_bottom.iloc[1]["n"]

    n_top_1 = DBR_top.iloc[0]["n"]
    n_top_2 = DBR_top.iloc[1]["n"]

    n_cavity = VCSEL.iloc[index_cavity]["n"]
    n_substrate = VCSEL.iloc[0]["n"]
    n_air = VCSEL.iloc[-1]["n"]

    return (
        n_bottom_1,
        n_bottom_2,
        N_bottom,
        n_top_1,
        n_top_2,
        N_top,
        n_cavity,
        n_substrate,
        n_air,
    )


def cavity_wavelength_temperature_correction(
    target_wavelength, T, temperature_coefficent=0.061e-9
):
    """
    Palik - Handbook of Optical
    Constants of Solids, p. 432:
    GaAs (1/n)(dn/dT) = (4.5 +- 0.2) * 1e-5 [1/C]
    Valid for 5-20um wavelength
    Approach over n(T) not feasable due to lack of data.

    Instead, model the known temperature dependency
    Hui Li p.34

    target_wavelength @ 300K

    """

    wavelength_T = target_wavelength + temperature_coefficent * (T - 300)

    return wavelength_T


def VCSEL_temperature_correction(
    VCSEL, target_wavelength, T, temperature_coefficent=0.061e-9
):

    VCSEL_structure = get_VCSEL_structure(VCSEL)
    wavelength_T = cavity_wavelength_temperature_correction(
        target_wavelength, T, temperature_coefficent
    )

    VCSEL_T = build_VCSEL_structure(*VCSEL_structure, wavelength_T)

    return VCSEL_T, wavelength_T


def interpolate_structure(structure, position_resolution: int = 100):

    # reset index in case of cropped structures and create empty copy
    structure_index_reset = structure.reset_index(drop=True)
    structure_interpolated = structure_index_reset.copy()[
        :0
    ]  # get empty structure to fill

    for i in range(len(structure_index_reset)):

        name = structure_index_reset.loc[i]["name"]
        n = structure_index_reset.loc[i]["n"]
        position = structure_index_reset.loc[i]["position"]
        d = structure_index_reset.loc[i]["d"]

        position_interpolated = np.linspace(
            position, position + d - (d / position_resolution), position_resolution
        )
        resolution = np.gradient(position_interpolated)

        layer_interpolated = pd.DataFrame(
            {
                "name": [name] * len(position_interpolated),
                "n": [n] * len(position_interpolated),
                "d": resolution,
                "position": position_interpolated,
            }
        )

        structure_interpolated = pd.concat(
            [
                structure_interpolated,
                layer_interpolated,
            ],
            ignore_index=True,
        )

    return structure_interpolated


def flip_structure(structure):
    structure_flipped = structure.iloc[::-1]
    structure_flipped = structure_flipped.reset_index(drop=True)

    position = 0.0
    position_arr = []
    for i in range(len(structure_flipped)):
        position_arr.append(position)
        d = structure_flipped.iloc[i]["d"]
        position += d

    structure_flipped["position"] = position_arr

    return structure_flipped


def reset_position(structure: pd.DataFrame) -> pd.DataFrame:
    """Reset the position column of a structure DataFrame."""
    pos = 0
    pos_arr = []
    for i in range(len(structure)):
        pos_arr.append(pos)
        d = structure.iloc[i]["d"]
        pos += d
    structure["position"] = pos_arr
    return structure


def structure_embedding(structure, n_embedding, d_embedding, offset=0.0):

    cladding1 = pd.DataFrame(
        {
            "name": ["Embedding"],
            "n": [n_embedding],
            "d": [d_embedding + offset],
            "position": [0],
        }
    )

    cladding2 = pd.DataFrame(
        {
            "name": ["Embedding"],
            "n": [n_embedding],
            "d": [d_embedding - offset],
            "position": [0],
        }
    )

    structure_embedded = pd.concat(
        [
            cladding1,
            structure,
            cladding2,
        ],
        ignore_index=True,
    )

    structure_embedded = reset_position(structure_embedded)
    return structure_embedded


def VCSEL_embedding_active_region(VCSEL, active_region, d_embedding=0.0, offset=0.0):

    index_cavity = VCSEL.loc[(VCSEL["name"] == "Cavity")].index.values[0]
    n_cavity = VCSEL.at[index_cavity, "n"]

    cavity_with_active_region = structure_embedding(
        active_region, n_embedding=n_cavity, d_embedding=d_embedding / 2, offset=offset
    )

    VCSEL_modified = pd.concat(
        [
            VCSEL[:index_cavity],
            cavity_with_active_region,
            VCSEL[index_cavity + 1 :],
        ],
        ignore_index=True,
    )
    VCSEL_modified = reset_position(VCSEL_modified)
    return VCSEL_modified


def build_active_region(n_arr, d_arr, active_region_idx=None):

    if active_region_idx == None:
        active_region_idx = [0]
    name_arr = [
        (
            "Modelled_Region_" + str(i) + "_Active_Region"
            if i in active_region_idx
            else "Modelled_Region_" + str(i)
        )
        for i in range(len(n_arr))
    ]

    active_region = pd.DataFrame(
        {
            "name": name_arr,
            "n": n_arr,
            "d": d_arr,
            "position": [0] * len(n_arr),
        }
    )
    active_region = reset_position(active_region)

    return active_region
