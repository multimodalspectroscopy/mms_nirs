from typing import Union

import numpy as np


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
