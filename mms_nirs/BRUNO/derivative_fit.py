from enum import Enum, auto
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .model_types import ExtrapolatedBoundaryConditions, ZeroBoundaryConditions


class BoundaryType(Enum):
    ZBC = auto()
    EBC = auto()


class QuantityType(Enum):
    REFLECTANCE = auto()
    ATTENUATION = auto()
    ATTENUATION_SLOPE = auto()


def get_model(
    boundary_condition_type: BoundaryType,
    quantity: QuantityType,
    distance_max: Optional[float],
):
    model_choice = (boundary_condition_type, quantity)
    match model_choice:
        case (BoundaryType.ZBC, QuantityType.REFLECTANCE):
            model_function = ZeroBoundaryConditions.reflectance()
        case (BoundaryType.ZBC, QuantityType.ATTENUATION):
            model_function = ZeroBoundaryConditions.attenuation()
        case (BoundaryType.ZBC, QuantityType.ATTENUATION_SLOPE):
            model_function = ZeroBoundaryConditions.attenuation_slope(
                is_long_separation=(distance_max is not None)
            )
        case (BoundaryType.EBC, QuantityType.REFLECTANCE):
            model_function = ExtrapolatedBoundaryConditions.reflectance()
        case (BoundaryType.EBC, QuantityType.ATTENUATION):
            model_function = ExtrapolatedBoundaryConditions.attenuation()
        case (BoundaryType.EBC, QuantityType.ATTENUATION_SLOPE):
            model_function = ExtrapolatedBoundaryConditions.attenuation_slope(
                is_long_separation=(distance_max is not None)
            )
        case _:
            raise ValueError(
                f"Incorrect model type passed. Passed {model_choice} but \
                    should be of type (BoundaryType, QuantityType)"
            )

    return model_function


def derivative_fit(
    param: NDArray[np.float64],
    boundary_condition_type: BoundaryType,
    quantity: QuantityType,
    slope_diff: NDArray[np.float64],
    extinction: NDArray[np.float64],
    wavelengths: NDArray[np.float64],
    distance: float,
    distance_max: Optional[float] = None,
    wave_start: float = 710.0,
    wave_end: float = 900.0,
) -> np.floating:
    """Create objective function to fit derivative

    Args:
        param (NDArray[np.float64]): First is water fraction, second is
        HHb, third is HbO2, fourth and fifth are the scattering coefficients
        from the exponential model; a and b resp., from mu_s = a*lambda^(-b)
        where lambda is in micrometers.

        boundary_condition_type (BoundaryType): Boundary type e.g. Zero or
        extrapolated

        quantity (QuantityType): Quantity to obtain. One of [reflectance,
        attenuation, attenuation slope].

        slope_diff (NDArray[np.float64]): Differential of slope. See
        calc_values.py for derivation

        extinction (NDArray[np.float64]): Extinction co-efficients matrix,
        W x 4, first column wavelength, second HHb, third HbO2, fourth the
        absorption coeff of water.

        wavelengths (NDArray[np.float64]): W x 1 array of wavelengths

        distance (float): Source-detector separation. If separation between
        detectors is negligible this is rho. If non-negligible this is the
        minimum distance from source to detector.

        distance_max (Optional[float], optional): Maximal source-detector
        distance. Not needed it distance bwteen detectors is negligible.
        Defaults to None.

        wave_start (int, optional): Start of wavelength range fitting is
        performed on. Defaults to 710.

        wave_end (int, optional): End of wavelength range fitting is performed
        on. Defaults to 900.

    Raises:
        ValueError: ValueError for unable to find unique start and end
        wavelengths

    Returns:
        np.floating: sum of least square differences
    """
    start_idx = np.argwhere(wavelengths == wave_start)
    end_idx = np.argwhere(wavelengths == wave_end)

    if (start_idx.shape != (1, 1)) or (end_idx.shape != (1, 1)):
        raise ValueError("Couldn't find unique start and end wavelengths")

    start_idx = start_idx[0][0]
    end_idx = end_idx[0][0]

    water_fraction, hhb_fraction, hbo2_fraction, a, b = param

    mu_a = water_fraction * extinction[:, 3] + np.log(10) * (
        hhb_fraction * extinction[:, 1] + hbo2_fraction * extinction[:, 2]
    )
    mu_s = a * (wavelengths * 0.001) ** (-b)

    rho = distance

    d_s, d_l = distance, distance_max

    model_function = get_model(boundary_condition_type, quantity, distance_max)
    if distance_max:
        slope_model_result = model_function(mu_s, mu_a, d_s, d_l)
    else:
        slope_model_result = model_function(mu_s, mu_a, rho)

    slope_model_diff = np.diff(slope_model_result, n=1)

    difference = (
        slope_model_diff[start_idx : end_idx + 1]
        - slope_diff[start_idx : end_idx + 1]
    )

    least_square = np.sum(difference**2)
    return least_square
