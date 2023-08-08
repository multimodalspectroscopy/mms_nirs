import numpy.testing as npt
import pytest

from mms_nirs.utils import calc_dpf


class TestDpf:
    @pytest.mark.parametrize(
        "mu_s,mu_a,d",
        [(3.0, 4.0, 5.0), (3.0, 4, 5.0), (3, 4.0, 5.0), (3, 4, 5)],
    )
    def test_calc_dpf(self, mu_s, mu_a, d):
        expected = 0.6979759

        actual = calc_dpf(mu_s, mu_a, d)

        npt.assert_approx_equal(actual, expected)
