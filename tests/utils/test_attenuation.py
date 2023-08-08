import numpy as np
import numpy.testing as npt
import pytest

from mms_nirs.utils.attenuation import (
    calc_attenuation_slope,
    calc_attenuation_spectra,
)


@pytest.fixture
def mock_ref_spectra():
    return np.array([1, 2, 3, 4])


@pytest.fixture
def mock_attenuation():
    return np.array(
        [
            [[2, 2, 2], [4, 4, 4]],
            [[4, 4, 4], [8, 8, 8]],
            [[6, 6, 6], [12, 12, 12]],
            [[8, 8, 8], [16, 16, 16]],
        ]
    )


class TestAttenuation:
    def test_single_intensity_spectrum(self, mock_ref_spectra):
        intensities = np.array([10, 20, 30, 40])

        expected = np.array([[-1.0, -1.0, -1.0, -1.0]])

        actual = calc_attenuation_spectra(intensities, mock_ref_spectra)

        npt.assert_array_almost_equal(actual, expected)

    def test_multiple_intensity_spectra(self, mock_ref_spectra):
        intensities = np.array([[10, 20, 30, 40], [20, 40, 60, 80]])

        expected = np.array(
            [
                [-1.0, -1.0, -1.0, -1.0],
                [-1.30103, -1.30103, -1.30103, -1.30103],
            ]
        )

        actual = calc_attenuation_spectra(intensities, mock_ref_spectra)

        npt.assert_array_almost_equal(actual, expected)


class TestAttenuationSlope:
    def test_attenuation_(self, mock_attenuation):
        with pytest.raises(ValueError):
            distances = np.array([1, 2, 3])
            calc_attenuation_slope(mock_attenuation, distances)

    def test_attenuation_slope(self, mock_attenuation):
        distances = np.array([1, 2, 3, 4])

        expected = np.array([[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]])

        actual = calc_attenuation_slope(mock_attenuation, distances)

        npt.assert_array_almost_equal(expected, actual)
