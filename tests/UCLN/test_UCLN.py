import pytest
from mms_nirs.UCLN import UCLN, UCLNConstants
from mms_nirs.UCLN import DefaultValues
import numpy as np
from numpy import testing as npt
import os

TEST_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

ROOT_DIR = os.path.dirname(TEST_DIR)


@pytest.fixture
def defaults() -> DefaultValues:
    return DefaultValues()


# We limit the number of values we use from the defaults to the number of
# columns in the spectra


@pytest.fixture
def extinction_coefficients(defaults: DefaultValues) -> np.ndarray:
    return defaults.extinction_coefficients


@pytest.fixture
def wavelength_dependency(defaults: DefaultValues) -> np.ndarray:
    return defaults.wavelength_dependency


@pytest.fixture
def spectra_wavelengths() -> np.ndarray:
    spectra_wavelengths: np.ndarray = np.genfromtxt(
        fname=TEST_DIR + "/test_data/UCLN/wavelengths.csv", delimiter=","
    )
    return spectra_wavelengths


@pytest.fixture
def spectra() -> np.ndarray:
    spectra: np.ndarray = np.genfromtxt(
        fname=TEST_DIR + "/test_data/UCLN/test_spectra.csv", delimiter=","
    )
    return spectra


@pytest.fixture
def true_conc() -> np.ndarray:
    conc: np.ndarray = np.genfromtxt(
        fname=TEST_DIR + "/test_data/UCLN/test_conc.csv", delimiter=","
    )
    return conc


@pytest.fixture
def ucln_constants(
    extinction_coefficients, wavelength_dependency
) -> UCLNConstants:
    return UCLNConstants(
        extinction_coefficients=extinction_coefficients,
        wavelength_dependency_of_pathlength=wavelength_dependency,
        optode_dist=3,
        dpf_type="baby_head",
        wavelengths=(780, 900),
    )


def test_calculate_concentrations(
    ucln_constants, spectra, spectra_wavelengths, true_conc
) -> None:
    ucln = UCLN(ucln_constants)

    conc = ucln.calc_concentrations(spectra, spectra_wavelengths)

    assert conc is not None, "Concentration returned a none value"
    npt.assert_almost_equal(
        conc, true_conc, err_msg="Concentration doesn't match expected value"
    )
