import numpy as np


def calc_dpf(mu_s: float, mu_a: float, d: float) -> float:
    """Calculate differential pathlength factor from mu_a and mu_s

    Taken from https://doi.org/10.1117/1.JBO.18.10.105004

    Args:
        mu_s (float): reduced scattering coefficient
        mu_a (float): absorption coefficient
        d (float): source-detector distance

    Returns:
        float: differential pathlength factor
    """

    return (
        0.5
        * np.sqrt(3 * mu_s / mu_a)
        * (1 - (1 / (1 + np.sqrt(d * 3 * mu_a * mu_s))))
    )
