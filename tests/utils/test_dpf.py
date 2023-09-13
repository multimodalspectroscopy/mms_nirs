import numpy as np
import numpy.testing as npt
import pytest

from mms_nirs.utils.dpf import calc_dpf, calc_mua, calc_mus


class TestDpf:
    @pytest.mark.parametrize(
        "mu_s,mu_a,d",
        [(3.0, 4.0, 5.0), (3.0, 4, 5.0), (3, 4.0, 5.0), (3, 4, 5)],
    )
    def test_calc_dpf_float(self, mu_s, mu_a, d):
        expected = 0.6979759

        actual = calc_dpf(mu_s, mu_a, d)

        npt.assert_approx_equal(actual, expected)

    @pytest.mark.parametrize(
        "mu_s,mu_a,d",
        [(3.0, 4.0, 5.0), (3.0, 4, 5.0), (3, 4.0, 5.0), (3, 4, 5)],
    )
    def test_calc_dpf_array(self, mu_s, mu_a, d):
        expected = np.array([0.6979759, 0.6979759])

        mu_s_arr = np.array([mu_s] * 2)
        mu_a_arr = np.array([mu_a] * 2)

        actual = calc_dpf(mu_s_arr, mu_a_arr, d)

        npt.assert_array_almost_equal(actual, expected)


class TestMua:
    def test_calc_mua(self):
        expected = np.array(
            [
                [5.403878, 5.403878, 5.403878],
                [5.484136, 5.484136, 5.484136],
                [5.864395, 5.864395, 5.864395],
            ]
        )

        water_frac = np.array([0.65, 0.6, 0.65])
        hhb = np.array([0.5, 0.4, 0.3])
        hbo2 = np.array([0.5, 0.6, 0.7])
        extinction_coefficients = np.array(
            [[0, 1, 2, 3], [1, 1, 2, 3], [2, 1, 2, 3]]
        )
        actual = calc_mua(
            water_frac=water_frac,
            hhb=hhb,
            hbo2=hbo2,
            extinction_coefficients=extinction_coefficients,
        )
        npt.assert_array_almost_equal(actual, expected)


class TestMus:
    def test_calc_mus(self):
        expected = np.array(
            [
                [3.922323, 3.902857, 3.883678],
                [3.081339, 3.069099, 3.057028],
                [2.270523, 2.263756, 2.257075],
            ]
        )

        a = np.array([0.65, 0.6, 0.65])
        b = np.array([0.5, 0.4, 0.3])
        wavelengths = np.array([100.0, 101.0, 102.0])
        actual = calc_mus(a=a, b=b, wavelengths=wavelengths)
        npt.assert_array_almost_equal(actual, expected)
