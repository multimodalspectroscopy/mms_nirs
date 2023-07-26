import numpy as np
import numpy.testing as npt

from mms_nirs.BRUNO.fminsearchbnd import fminsearchbnd


def rosen(x):
    return (1 - x[0]) ** 2 + 105 * (x[1] - x[0] ** 2) ** 2


class TestFminsearchbnd:
    def test_unconstrained(self):
        expected_x = np.array([0.999977949278684, 0.999953540900379])
        result = fminsearchbnd(rosen, [3, 3])
        assert result["success"]
        npt.assert_array_almost_equal(result["x"], expected_x)

    def test_only_LB(self):
        expected_x = np.array([2.000000000528625, 3.999991976788512])
        result = fminsearchbnd(rosen, [3, 3], [2, 2], [])
        assert result["success"]
        npt.assert_array_almost_equal(result["x"], expected_x)

    def test_constrained(self):
        expected_x = np.array([2.000000000141436, 2.999999999705527])
        result = fminsearchbnd(rosen, [3, 3], [2, 2], [np.inf, 3.0])
        assert result["success"]
        npt.assert_array_almost_equal(result["x"], expected_x)
