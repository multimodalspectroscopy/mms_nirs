from typing import Union

import numpy as np


def calc_mua(
    water_frac: np.ndarray,
    hhb: np.ndarray,
    hbo2: np.ndarray,
    extinction_coefficients: np.ndarray,
) -> np.ndarray:
    """Calculate absorption coefficient

    Args:
        water_frac (np.ndarray): Water fraction
        hhb (np.ndarray): Deoxy Hb conc
        hbo2 (np.ndarray): Oxy Hb conc
        extinction_coefficients (np.ndarray): Extinction coefficients

    Returns:
        np.ndarray: Array of mu_a for each wavelength and timepoint
    """
    mua = np.outer(water_frac, extinction_coefficients[:, 3]) + np.log(10) * (
        np.outer(hhb, extinction_coefficients[:, 1])
        + np.outer(hbo2, extinction_coefficients[:, 2])
    )

    return mua


def calc_mus(
    a: np.ndarray, b: np.ndarray, wavelengths: np.ndarray
) -> np.ndarray:
    """Calculate scattering coefficient

    Args:
        a (np.ndarray): parameter a in scattering coefficient calculation
        b (np.ndarray): parameter b in scattering coefficient calculation
        wavelengths (np.ndarray): Wavelengths to calculate at

    Returns:
        np.ndarray: Array of mu_s for each wavelength and timepoint
    """
    mus = np.power(
        np.outer(a, (wavelengths * 0.001)).T,
        -b,
    ).T

    return mus


def calc_dpf(
    mu_s: Union[float, np.ndarray], mu_a: Union[float, np.ndarray], d: float
) -> Union[float, np.ndarray]:
    """Calculate differential pathlength factor from mu_a and mu_s

    If arrays are passed, output is an array. If floats are passed
    output is a float

    Taken from https://doi.org/10.1117/1.JBO.18.10.105004

    Args:
        mu_s (float | np.ndarray): reduced scattering coefficient
        mu_a (float | np.ndarray): absorption coefficient
        d (float): source-detector distance

    Returns:
        float | np.ndarray: differential pathlength factor
    """

    return (
        0.5
        * np.sqrt(3 * mu_s / mu_a)
        * (1 - (1 / (1 + np.sqrt(d * 3 * mu_a * mu_s))))
    )
