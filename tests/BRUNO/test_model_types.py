import numpy as np
import numpy.testing as npt

from mms_nirs.BRUNO import (
    ExtrapolatedBoundaryConditions,
    ZeroBoundaryConditions,
)

mu_a = np.array([1.0, 2.0, 3.0])
mu_s = np.array([2.0, 3.0, 4.0])
rho = 3.0
d_s = 15.0
d_l = 30.0


class TestExtrapolatedBoundaryConditions:
    def test_reflectance(self):
        expected = np.array([1.394e-05, 8.988e-08, 5.504e-10])
        actual = ExtrapolatedBoundaryConditions.reflectance()(mu_s, mu_a, rho)
        npt.assert_almost_equal(actual, expected)

    def test_attenuation(self):
        expected = np.array([4.8558, 7.0464, 9.2593])
        actual = ExtrapolatedBoundaryConditions.attenuation()(mu_s, mu_a, rho)
        npt.assert_almost_equal(actual, expected, decimal=4)

    def test_attenuation_slope_short_separation(self):
        expected = np.array([1.2686, 2.0576, 2.8351])
        actual = ExtrapolatedBoundaryConditions.attenuation_slope()(
            mu_s, mu_a, rho
        )
        npt.assert_almost_equal(actual, expected, decimal=4)

    def test_attenuation_slope_long_separation(self):
        expected = np.array([1.1012, 1.8809, 2.6446])
        actual = ExtrapolatedBoundaryConditions.attenuation_slope(
            is_long_separation=True
        )(mu_s, mu_a, d_s, d_l)
        npt.assert_almost_equal(actual, expected, decimal=4)


class TestZeroBoundaryConditions:
    def test_reflectance(self):
        expected = np.array([1.394e-05, 7.420e-08, 4.040e-10])
        actual = ZeroBoundaryConditions.reflectance()(mu_s, mu_a, rho)
        npt.assert_almost_equal(actual, expected)

    def test_attenuation(self):
        expected = np.array([4.8558, 7.1296, 9.3936])
        actual = ZeroBoundaryConditions.attenuation()(mu_s, mu_a, rho)
        npt.assert_almost_equal(actual, expected, decimal=4)

    def test_attenuation_slope_short_separation(self):
        expected = np.array([1.3533, 2.1321, 2.8953])
        actual = ZeroBoundaryConditions.attenuation_slope()(mu_s, mu_a, rho)
        npt.assert_almost_equal(actual, expected, decimal=4)

    def test_attenuation_slope_long_separation(self):
        expected = np.array([1.1039, 1.8827, 2.6459])
        actual = ZeroBoundaryConditions.attenuation_slope(
            is_long_separation=True
        )(mu_s, mu_a, d_s, d_l)
        npt.assert_almost_equal(actual, expected, decimal=4)
