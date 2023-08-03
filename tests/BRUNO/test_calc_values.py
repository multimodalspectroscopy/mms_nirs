from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from mms_nirs.BRUNO.calc_values import calc_values, smooth
from mms_nirs.BRUNO.derivative_fit import BoundaryType

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
def mock_slope():
    return np.genfromtxt(FIXTURE_DIR / "slope.csv", delimiter=",")


@pytest.fixture
def mock_boundaries():
    return np.array(
        [
            [1.0, 20.0, 20.0, 1.0, 3.0],
            [0.970000000000000, 0.0, 0.0, 0.0, 0.0],
            [1.0, 40.0, 40.0, 2.0, 4.0],
        ]
    )


@pytest.fixture
def function_arguments(
    mock_extinctions, mock_wavelengths, mock_boundaries, mock_slope
):
    return {
        "slope": mock_slope,
        "extinction": mock_extinctions,
        "wavelengths": mock_wavelengths,
        "boundaries": mock_boundaries,
        "distance": 22.5,
    }


class TestCalcValues:
    def test_produces_correct_value_for_close_separation_ZBC(
        self, function_arguments
    ):
        expected_stO2 = 84.034715681079630
        expected_coefficients = np.array(
            [
                0.999231368648997,
                3.884967953622380,
                20.448879637292890,
                0.132585802300888,
                2.555661423244912,
            ]
        )

        expected_score = 0.15233402419854875

        (
            stO2,
            coefficients,
            residual,
            residual_norm,
            sum_residual,
            score,
        ) = calc_values(
            boundary_condition_type=BoundaryType.ZBC,
            **function_arguments,
        )
        npt.assert_array_almost_equal(
            coefficients, expected_coefficients, decimal=5.0
        )
        npt.assert_approx_equal(stO2, expected_stO2)
        npt.assert_approx_equal(score, expected_score)

    def test_produces_correct_value_for_far_separation_ZBC(
        self, function_arguments
    ):
        expected_stO2 = 84.034715681079630
        expected_coefficients = np.array(
            [
                0.999231368648997,
                3.884967953622380,
                20.448879637292890,
                0.132585802300888,
                2.555661423244912,
            ]
        )
        expected_score = 33.85892581587063

        (
            stO2,
            coefficients,
            residual,
            residual_norm,
            sum_residual,
            score,
        ) = calc_values(
            boundary_condition_type=BoundaryType.ZBC,
            **function_arguments,
            distance_max=45.0,
        )
        npt.assert_array_almost_equal(
            coefficients, expected_coefficients, decimal=5.0
        )
        npt.assert_approx_equal(stO2, expected_stO2)
        npt.assert_approx_equal(score, expected_score)

    # We set the precision slightly lower on these ones as the EBC algo
    # differs ever so slightly in return value between Matlab and Scipy
    def test_produces_correct_value_for_close_separation_EBC(
        self, function_arguments
    ):
        expected_stO2 = 88.512150155916840
        expected_coefficients = np.array(
            [
                0.970000000000045,
                3.879809932233005,
                29.893350275212660,
                0.281392552581535,
                1.806582390291361,
            ]
        )

        expected_score = 0.27953102851704975

        (
            stO2,
            coefficients,
            residual,
            residual_norm,
            sum_residual,
            score,
        ) = calc_values(
            boundary_condition_type=BoundaryType.EBC,
            **function_arguments,
        )
        npt.assert_array_almost_equal(
            coefficients, expected_coefficients, decimal=4.0
        )
        npt.assert_approx_equal(stO2, expected_stO2)
        npt.assert_approx_equal(score, expected_score)

    def test_produces_correct_value_for_far_separation_EBC(
        self, function_arguments
    ):
        expected_stO2 = 87.030305470331020
        expected_coefficients = np.array(
            [
                0.970000000000008,
                3.884161902993422,
                26.063821020645683,
                0.228394781489393,
                1.968340719908830,
            ]
        )

        expected_score = 57.2088269575511

        (
            stO2,
            coefficients,
            residual,
            residual_norm,
            sum_residual,
            score,
        ) = calc_values(
            boundary_condition_type=BoundaryType.EBC,
            **function_arguments,
            distance_max=45.0,
        )
        npt.assert_array_almost_equal(
            coefficients, expected_coefficients, decimal=4.0
        )
        npt.assert_approx_equal(stO2, expected_stO2)
        npt.assert_approx_equal(score, expected_score)
