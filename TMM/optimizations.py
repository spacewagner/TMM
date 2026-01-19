import numpy as np
from scipy.signal import find_peaks, peak_widths, peak_prominences
from dataclasses import dataclass
import matplotlib.pyplot as plt

from TMM.structure_builder import VCSEL_embedding_active_region
from TMM.analysis import analyse_electrical_field


# %% Find cavity length for optimum confinement


def VCSEL_embedding_sweep(VCSEL, active_region, d_embedding_arr, target_wavelength):
    Gamma_z_arr = []
    Gamma_z_active_region_arr = []
    for d_embedding in d_embedding_arr:
        VCSEL_modified = VCSEL_embedding_active_region(
            VCSEL, active_region, d_embedding=d_embedding
        )

        results = analyse_electrical_field(
            VCSEL_modified,
            target_wavelength,
            Plot=False,
            Print=False,
        )
        # idx_embedding = VCSEL_modified.loc[VCSEL_modified["name"] == "Embedding"].index
        # plt.plot(
        #     results.field_positions_arr[
        #         idx_embedding[0]
        #         * position_resolution : (idx_embedding[1] + 1)
        #         * position_resolution
        #     ],
        #     np.abs(
        #         results.field_values_arr[
        #             idx_embedding[0]
        #             * position_resolution : (idx_embedding[1] + 1)
        #             * position_resolution
        #         ]
        #     )
        #     ** 2
        #     / np.max(np.abs(results.field_values_arr) ** 2),
        # )
        Gamma_z = results.Gamma_z
        Gamma_z_arr.append(Gamma_z)

        Gamma_z_active_region = results.Gamma_z_active_region
        Gamma_z_active_region_arr.append(Gamma_z_active_region)

    plt.plot(d_embedding_arr, Gamma_z_arr)
    plt.plot(d_embedding_arr, Gamma_z_active_region_arr)
    plt.show()
    return np.array(Gamma_z_arr), np.array(Gamma_z_active_region_arr)


def plot_VCSEL_embedding_sweep(results, plot_embedding_thickness=True):
    plt.plot(
        (results.d_embedding_arr + results.d_active_region) * 1e9,
        results.Gamma_z_arr,
        label="Cavity",
    )
    plt.plot(
        (results.d_embedding_arr + results.d_active_region) * 1e9,
        results.Gamma_z_active_region_arr,
        label="Active Region",
    )

    for pos, val in zip(results.d_optimum_arr, results.Gamma_z_optimum_arr):
        plt.plot((pos + results.d_active_region) * 1e9, val, marker="o", color="red")

    plt.xlabel("Cavity Length (nm)")
    plt.ylabel("Mode Confinement $\\Gamma_z$")
    plt.legend()

    if plot_embedding_thickness:
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


@dataclass
class ActiveRegionEmbeddingProperties:
    d_embedding_arr: np.ndarray
    Gamma_z_arr: np.ndarray
    Gamma_z_active_region_arr: np.ndarray

    d_optimum_arr: np.ndarray
    Gamma_z_optimum_arr: np.ndarray

    d_active_region: float


def optimize_embedding_thickness(
    VCSEL,
    active_region,
    target_wavelength,
    d_min,
    d_max,
    iter=10,
    plot_embedding_thickness=True,
):
    d_active_region = active_region["d"].sum()
    d_arr_resolution = 5e-9
    d_min = d_min
    d_max = d_max
    d_embedding_arr_arr = []
    Gamma_z_arr_arr = []
    Gamma_z_active_region_arr_arr = []
    d_embedding_arr = np.arange(d_min, d_max + 1e-12, d_arr_resolution)
    max_arr = []
    amount_of_iter = []
    for i in range(iter):

        Gamma_z_arr, Gamma_z_active_region_arr = VCSEL_embedding_sweep(
            VCSEL, active_region, d_embedding_arr, target_wavelength
        )

        d_embedding_arr_arr.append(d_embedding_arr)
        Gamma_z_arr_arr.append(Gamma_z_arr)
        Gamma_z_active_region_arr_arr.append(Gamma_z_active_region_arr)

        peaks, properties = find_peaks(
            Gamma_z_arr,
            height=0,
            threshold=None,
            distance=1,
        )
        d_optimum_arr = d_embedding_arr[peaks]
        Gamma_z_optimum_arr = Gamma_z_arr[peaks]

        max_arr.append(np.sum(Gamma_z_optimum_arr))
        amount_of_iter.append(i + 1)

        plt.plot(amount_of_iter, max_arr, marker=".")
        plt.xlabel("N Iterations")
        plt.ylabel("Peak Amplitude Sum")
        plt.show()

        # optimization stops if there was no improvement in the last 2 iterations greater than 0.1%
        if i > 1 and (max_arr[i] / max_arr[i - 2]) < 1.001:
            break

        if len(d_optimum_arr) == 0:
            break
        d_embedding_arr = np.concatenate(
            [
                np.arange(
                    d_optimum_arr[i] - d_arr_resolution,
                    d_optimum_arr[i] + d_arr_resolution + 1e-12,
                    d_arr_resolution / 2,
                )
                for i in range(len(d_optimum_arr))
            ]
        )
        d_arr_resolution /= 2

    d_embedding_arr_combined = []
    for d_embedding_arr in d_embedding_arr_arr:
        d_embedding_arr_combined += list(d_embedding_arr)
    Gamma_z_arr_combined = []
    for Gamma_z_arr in Gamma_z_arr_arr:
        Gamma_z_arr_combined += list(Gamma_z_arr)
    Gamma_z_active_region_arr_combined = []
    for Gamma_z_active_region_arr in Gamma_z_active_region_arr_arr:
        Gamma_z_active_region_arr_combined += list(Gamma_z_active_region_arr)

    pairs = sorted(
        zip(
            d_embedding_arr_combined,
            Gamma_z_arr_combined,
            Gamma_z_active_region_arr_combined,
        )
    )
    d_embedding_arr_sorted, Gamma_z_arr_sorted, Gamma_z_active_region_arr_sorted = zip(
        *pairs
    )

    d_embedding_arr = np.array(d_embedding_arr_sorted)
    Gamma_z_arr = np.array(Gamma_z_arr_sorted)
    Gamma_z_active_region_arr = np.array(Gamma_z_active_region_arr_sorted)

    peaks, properties = find_peaks(
        Gamma_z_arr,
        height=0,
        threshold=None,
        distance=1,
    )
    d_optimum_arr = d_embedding_arr[peaks]
    Gamma_z_optimum_arr = Gamma_z_arr[peaks]
    results = ActiveRegionEmbeddingProperties(
        d_embedding_arr=d_embedding_arr,
        Gamma_z_arr=Gamma_z_arr,
        Gamma_z_active_region_arr=Gamma_z_active_region_arr,
        d_optimum_arr=d_optimum_arr,
        Gamma_z_optimum_arr=Gamma_z_optimum_arr,
        d_active_region=d_active_region,
    )

    plot_VCSEL_embedding_sweep(results)

    return results
