import numpy as np
import numpy.testing as npt
import pytest

from mms_nirs.utils.dpf import calc_dpf


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
