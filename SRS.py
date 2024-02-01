import numpy as np
from functools import partial

def calc_k_mua(
    slope: np.ndarray,
    wavelengths: np.ndarray,
    min_distance: float,
    max_distance: float,
):
    dist_ratio = max_distance / min_distance
    dist_diff = max_distance - min_distance

    h = 6.3e-4
    return ((np.log(10) * slope) - (2 * np.log(dist_ratio) / dist_diff)) / (
        3 * (1 - h * wavelengths)
    )


def calc_conc(k_mua, ext_coeffs_inv):
    return np.matmul(ext_coeffs_inv, k_mua)


def calc_sto2(conc):
    oxy = conc[0]
    deoxy = conc[1]

    return (oxy / (oxy + deoxy)) * 100



def srs_values(
    slope: np.ndarray,
    wavelengths: np.ndarray,
    ext_coeffs_inv: np.ndarray,
    min_distance: float,
    max_distance: float,
):
    #Calculate k_mua
    k_mua = calc_k_mua(slope = slope,wavelengths = wavelengths, min_distance = min_distance, max_distance = max_distance)

    #Calculate concentrations
    C = calc_conc(k_mua,ext_coeffs_inv)

    #Calculate StO2
    StO2 = calc_sto2(C)

    return C,StO2,k_mua


