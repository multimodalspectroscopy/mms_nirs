from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from mms_nirs.BRUNO.calc_values import smooth
from mms_nirs.BRUNO.derivative_fit import (
    BoundaryType,
    QuantityType,
    derivative_fit,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def mock_extinctions():
    return np.genfromtxt(FIXTURE_DIR / "extinctions.csv", delimiter=",")


@pytest.fixture
def mock_wavelengths():
    return np.genfromtxt(FIXTURE_DIR / "wavelengths.csv", delimiter=",")


@pytest.fixture
def mock_attenuations():
    return np.genfromtxt(FIXTURE_DIR / "attenuations.csv", delimiter=",")


@pytest.fixture
def function_arguments(mock_extinctions, mock_wavelengths):
    # Slope taken from generate_example matlab file
    slope = np.genfromtxt(FIXTURE_DIR / "slope.csv", delimiter=",")
    smoothed_slope = smooth(slope, 5)
    slope_1st_diff = np.diff(smoothed_slope)
    return {
        "slope_diff": slope_1st_diff,
        "extinction": mock_extinctions,
        "wavelengths": mock_wavelengths,
        "distance": 22.5,
    }


class TestDerivativeFit:
    def test_produces_correct_value_for_close_separation_ZBC(
        self, function_arguments
    ):
        expected = 2.325640442641589e-05

        param = np.array([1.0, 20.0, 20.0, 1.0, 3.0])

        actual = derivative_fit(
            param,
            boundary_condition_type=BoundaryType.ZBC,
            quantity=QuantityType.ATTENUATION_SLOPE,
            **function_arguments,
        )
        npt.assert_approx_equal(actual, expected)

    def test_produces_correct_value_for_far_separation_ZBC(
        self, function_arguments
    ):
        expected = 2.325640442641589e-05

        param = np.array([1.0, 20.0, 20.0, 1.0, 3.0])

        actual = derivative_fit(
            param,
            boundary_condition_type=BoundaryType.ZBC,
            quantity=QuantityType.ATTENUATION_SLOPE,
            distance_max=45,
            **function_arguments,
        )
        npt.assert_approx_equal(actual, expected)

    def test_produces_correct_value_for_close_separation_EBC(
        self, function_arguments
    ):
        expected = 2.236493937675748e-05

        param = np.array([1.0, 20.0, 20.0, 1.0, 3.0])

        actual = derivative_fit(
            param,
            boundary_condition_type=BoundaryType.EBC,
            quantity=QuantityType.ATTENUATION_SLOPE,
            **function_arguments,
        )
        npt.assert_approx_equal(actual, expected)

    def test_produces_correct_value_for_far_separation_EBC(
        self, function_arguments
    ):
        expected = 2.2750546451957276e-05

        param = np.array([1.0, 20.0, 20.0, 1.0, 3.0])

        actual = derivative_fit(
            param,
            boundary_condition_type=BoundaryType.EBC,
            quantity=QuantityType.ATTENUATION_SLOPE,
            distance_max=45,
            **function_arguments,
        )
        npt.assert_approx_equal(actual, expected)

    def test_raises_error_on_non_unique_wavelengths(
        self, mock_wavelengths, function_arguments
    ):
        # Repeat wavelength 710nm
        mock_wavelengths[7] = mock_wavelengths[6]
        param = np.array([1.0, 20.0, 20.0, 1.0, 3.0])

        with pytest.raises(ValueError):
            derivative_fit(
                param,
                boundary_condition_type=BoundaryType.ZBC,
                quantity=QuantityType.ATTENUATION_SLOPE,
                **function_arguments,
            )


# class TestGetModel:
#     #TO DO
