import numpy as np
from numpy.linalg import lstsq
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


def calc_attenuation_slope(
    attenuation_spectra: NDArray, source_detector_distances: NDArray
) -> NDArray:
    """Calculate the attenuation slope via linear regression.

    Calculate attenuation slope at each of `T` timepoints for a spectra with
    `N` wavelengths and `k` source-detector distances

    Args:
        attenuation_spectra (NDArray): Matrix of attenuation spectra of shape
        `T`x`N`x`k`
        source_detector_distances (NDArray): List of source-detector distances.
        Should be of length `k`

    Returns:
        NDArray: Matrix of attenuation slope for each timepoint. Shape `T`x`N`
    """

    k, T, N = attenuation_spectra.shape

    if len(source_detector_distances) != k:
        raise ValueError(
            f"Mismatch between numbers of distances and layers of attenuation\
                matrix.\n\
                Got {len(source_detector_distances)} and {k} respectively."
        )

    # Solve for slope of form y = mx + c = Ap
    def get_slope(attenuations: NDArray, distances: NDArray):
        A = np.vstack([distances, np.ones(len(distances))]).T
        m, _ = lstsq(A, attenuations, rcond=None)[0]
        return m  # type: ignore

    return np.apply_along_axis(
        get_slope, 0, attenuation_spectra, source_detector_distances
    )
