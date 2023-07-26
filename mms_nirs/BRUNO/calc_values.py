from typing import Optional

import numpy as np

from .derivative_fit import (
    BoundaryType,
    QuantityType,
    derivative_fit,
    get_model,
)
from .fminsearchbnd import fminsearchbnd


def smooth(a, span):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a, np.ones(span, dtype=int), "valid") / span
    r = np.arange(1, span - 1, 2)
    start = np.cumsum(a[: span - 1])[::2] / r
    stop = (np.cumsum(a[:-span:-1])[::2] / r)[::-1]
    return np.concatenate((start, out0, stop))


def calc_values(
    slope: np.ndarray,
    extinction: np.ndarray,
    wavelengths: np.ndarray,
    boundaries: np.ndarray,
    boundary_condition_type: BoundaryType,
    distance: float,
    distance_max: Optional[float] = None,
):
    start = boundaries[0]
    LB = boundaries[1]
    UB = boundaries[2]

    slope_1stdiff = np.diff(smooth(slope, 5))
    # Set wavelength range for the fitting
    wave_start = 710
    wave_end = 900

    # Setting fitting options
    options = {
        "disp": False,
        "maxiter": 200000,
        "maxfev": 200000,
        "xatol": 1e-10,
        "fatol": 1e-10,
    }

    result = fminsearchbnd(
        derivative_fit,
        x0=start,
        LB=LB,
        UB=UB,
        func_args=(
            boundary_condition_type,
            QuantityType.ATTENUATION_SLOPE,
            slope_1stdiff,
            extinction,
            wavelengths,
            distance,
            distance_max,
            wave_start,
            wave_end,
        ),
        options=options,
        tol=1e-10,
    )

    if result["success"]:
        coefficients = result["x"]
    else:
        raise RuntimeError("Failed to solve for coefficients.")

    mua = coefficients[0] * extinction[:, 3] + np.log(10) * (
        coefficients[1] * extinction[:, 1] + coefficients[2] * extinction[:, 2]
    )
    mus = coefficients[3] * (wavelengths * 0.001) ** (-coefficients[4])

    model_function = get_model(
        boundary_condition_type, QuantityType.ATTENUATION_SLOPE, distance_max
    )
    if distance_max:
        model_result = model_function(distance_max, distance, mua, mus)
    else:
        model_result = model_function(mua, mus, distance)

    model_1stdiff = np.diff(model_result, n=1)

    stO2 = coefficients[2] / (coefficients[1] + coefficients[2]) * 100

    residual = (model_1stdiff - slope_1stdiff) ** 2
    residual_norm = (
        model_1stdiff / np.max(model_1stdiff)
        - slope_1stdiff / np.max(model_1stdiff)
    ) ** 2
    sum_residual = np.sum(residual)

    index_HHb = np.arange(
        np.where(wavelengths == 750)[0][0],
        np.where(wavelengths == 770)[0][0] + 1,
    )
    index_water = np.arange(
        np.where(wavelengths == 825)[0][0],
        np.where(wavelengths == 840)[0][0] + 1,
    )
    sum_hhb_residuals = np.sum(residual_norm[index_HHb])
    sum_water_residuals = np.sum(residual_norm[index_water])
    model_range = np.max(
        model_1stdiff[6:197] / np.max(model_1stdiff[6:197])
    ) - np.min(model_1stdiff[6:197] / np.max(model_1stdiff[6:197]))
    score = sum_hhb_residuals * sum_water_residuals / model_range

    return stO2, coefficients, residual, residual_norm, sum_residual, score
