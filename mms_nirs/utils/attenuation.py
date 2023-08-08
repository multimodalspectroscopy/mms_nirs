import numpy as np
from numpy.typing import NDArray


def calc_attenuation_spectra(
    intensity_spectra: NDArray, ref_spectra: NDArray
) -> NDArray:
    """Calculate attenuation spectra from intensities

    Args:
        intensity_spectra (NDArray): Intensity spectra
        ref_spectra (NDArray): Re:w
        ference spectrum

    Returns:
        NDArray: Calculated attenuation spectra. Will convert 1D array to 2D
    """

    if intensity_spectra.ndim == 1:
        intensity_spectra = intensity_spectra.reshape(
            -1, len(intensity_spectra)
        )
    return np.log10(np.divide(ref_spectra, intensity_spectra))
