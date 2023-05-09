import pytest
from mms_nirs.MBL import MBL, MBLConstants
from mms_nirs.DefaultValues import DefaultValues
import numpy as np
from numpy import testing as npt
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


@pytest.fixture
def spectra() -> np.ndarray:
    return np.array(
        [
            [1, 1, 2, 2, 3, 3, 3, 2, 2, 3, 3, 2, 2, 1],
            [1, 2, 2, 2, 2, 3, 3, 2, 2, 2, 1, 1, 1, 1],
            [1, 2, 2, 3, 3, 3, 3, 2, 2, 3, 2, 2, 2, 1],
            [1, 1, 1, 2, 2, 3, 3, 2, 2, 2, 1, 1, 1, 1],
            [1, 1, 1, 2, 1, 2, 3, 2, 2, 1, 1, 2, 2, 1],
        ]
    )


@pytest.fixture
def defaults() -> DefaultValues:
    default_fpath = ROOT_DIR + "/mms_nirs/defaults.csv"
    return DefaultValues(csv_file=default_fpath)


# We limit the number of values we use from the defaults to the number of
# columns in the spectra


@pytest.fixture
def extinction_coefficients(
    defaults: DefaultValues, spectra: np.ndarray
) -> np.ndarray:
    n_wl = spectra.shape[1]
    return defaults.extinction_coefficients[:n_wl]


@pytest.fixture
def wavelength_dependency(
    defaults: DefaultValues, spectra: np.ndarray
) -> np.ndarray:
    n_wl = spectra.shape[1]
    return defaults.wavelength_dependency[:n_wl]


@pytest.fixture
def spectra_wavelengths(
    defaults: DefaultValues, spectra: np.ndarray
) -> np.ndarray:
    n_wl = spectra.shape[1]
    return defaults.spectra_wavelengths[:n_wl]


@pytest.fixture
def mbl_constants(
    extinction_coefficients, wavelength_dependency
) -> MBLConstants:
    return MBLConstants(
        extinction_coefficients,
        wavelength_dependency,
        optode_dist=3,
        dpf_type="adult_head",
        wavelengths=(780, 793),
    )


def test_calculate_concentrations(
    mbl_constants, spectra, spectra_wavelengths
) -> None:
    mbl = MBL(mbl_constants)

    conc = mbl.calc_concentrations(spectra, spectra_wavelengths)

    true_conc: np.ndarray = np.array(
        [
            [0, 0, 0],
            [0.265095297287314, -0.0340896301818841, -0.0794803591724266],
            [-0.217963488242189, -0.0653206231358632, 0.109851244531198],
            [0.692103669039122, 0.0630682559053555, -0.280351607759156],
            [-0.921785501953610, -0.134709818934567, 0.404545406828108],
        ]
    )

    assert conc is not None, "Calculated concentration returned None"
    npt.assert_almost_equal(true_conc, conc)
