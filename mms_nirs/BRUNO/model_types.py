import sympy
from sympy import lambdify


# ~~~~~~~~~~~~~~ Define unresolved symbolic equations ~~~~~~~~~~~~~~~~~~~~~~~ #
def zbc_reflectance(mu_s, mu_a, rho):
    z0 = 1 / mu_s
    mueff = sympy.sqrt(3 * mu_a * mu_s)
    return (z0 * mueff * sympy.exp(-mueff * rho)) / (2 * sympy.pi * rho**2)


def zbc_attenuation(mu_s, mu_a, rho):
    z0 = 1 / mu_s
    mueff = sympy.sqrt(3 * mu_a * mu_s)
    return (
        (mueff * rho)
        + (2.0 * sympy.log(rho))  # type: ignore
        - sympy.log((z0 * mueff) / (2 * sympy.pi))
    ) / sympy.log(10)


def zbc_attenuation_slope_short_separation(mu_s, mu_a, rho):
    return (sympy.sqrt(3 * mu_s * mu_a) + 2 / rho) / sympy.log(10)


def zbc_attenuation_slope_long_separation(mu_s, mu_a, d_s, d_l):
    return (
        sympy.sqrt(3 * mu_s * mu_a)
        + 2 * ((sympy.log(d_l / d_s)) / (d_l - d_s))
    ) / sympy.log(10)


# ~~~~~~~~~~~~~ End unresolve symbolic equations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class ZeroBoundaryConditions:
    @staticmethod
    def reflectance():
        mu_s, mu_a, rho = sympy.symbols("mu_s mu_a rho")
        return lambdify(
            [mu_s, mu_a, rho], zbc_reflectance(mu_s, mu_a, rho), "numpy"
        )

    @staticmethod
    def attenuation():
        mu_s, mu_a, rho = sympy.symbols("mu_s mu_a rho")
        return lambdify(
            [mu_s, mu_a, rho], zbc_attenuation(mu_s, mu_a, rho), "numpy"
        )

    @staticmethod
    def attenuation_slope(is_long_separation: bool = False):
        mu_s, mu_a, rho, d_s, d_l = sympy.symbols("mu_s mu_a rho d_s d_l")

        if is_long_separation:
            return lambdify(
                [mu_s, mu_a, d_s, d_l],
                zbc_attenuation_slope_long_separation(mu_s, mu_a, d_s, d_l),
                "numpy",
            )

        return lambdify(
            [mu_s, mu_a, rho],
            zbc_attenuation_slope_short_separation(mu_s, mu_a, rho),
            "numpy",
        )


# ~~~~~~~~~~~~~~ Define unresolved symbolic equations ~~~~~~~~~~~~~~~~~~~~~~~ #
def ebc_reflectance(mu_s, mu_a, rho):
    z0 = 1 / mu_s
    mueff = sympy.sqrt(3 * mu_a * mu_s)
    D = 1.0 / (3 * (mu_a + mu_s))

    zb = (
        (1 + 0.493) / (1 - 0.493) * 2 * D
    )  # from Kienle(1997),j. Opt. Soc. Am. A 14:1.
    # Valid for biological tissue.
    r1sq = (
        rho**2
    )  # r1^2 = z0^2 + rho^2 but z0^2 is negligible compared to rho^2
    r2 = (z0 + 2 * zb) ** 2 + rho**2
    return (
        1
        / (4 * sympy.pi)
        * (
            z0
            * (mueff + 1.0 / sympy.sqrt(r1sq))  # type: ignore
            * (sympy.exp(-mueff * sympy.sqrt(r1sq)) / r1sq)  # type: ignore
            + (z0 + 2 * zb)
            * (mueff + 1.0 / sympy.sqrt(r2))  # type: ignore
            * (sympy.exp(-mueff * sympy.sqrt(r2)) / r2)  # type: ignore
        )
    )


def ebc_attenuation(mu_s, mu_a, rho):
    return -1.0 * sympy.log(
        ebc_reflectance(mu_s, mu_a, rho), 10
    )  # type: ignore


def ebc_attenuation_slope_short_separation(mu_s, mu_a, rho):
    return sympy.diff(ebc_attenuation(mu_s, mu_a, rho), rho)


def ebc_attenuation_slope_long_separation(mu_s, mu_a, d_s, d_l):
    return (
        ebc_attenuation(mu_s, mu_a, d_l) - ebc_attenuation(mu_s, mu_a, d_s)
    ) / (
        d_l - d_s
    )  # type: ignore


# ~~~~~~~~~~~~~ End unresolve symbolic equations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class ExtrapolatedBoundaryConditions:
    @staticmethod
    def reflectance():
        mu_s, mu_a, rho = sympy.symbols("mu_s mu_a rho")
        return lambdify(
            [mu_s, mu_a, rho], ebc_reflectance(mu_s, mu_a, rho), "numpy"
        )

    @staticmethod
    def attenuation():
        mu_s, mu_a, rho = sympy.symbols("mu_s mu_a rho")
        return lambdify(
            [mu_s, mu_a, rho], ebc_attenuation(mu_s, mu_a, rho), "numpy"
        )

    @staticmethod
    def attenuation_slope(is_long_separation: bool = False):
        mu_s, mu_a, rho, d_s, d_l = sympy.symbols("mu_s mu_a rho d_s d_l")

        if is_long_separation:
            return lambdify(
                [mu_s, mu_a, d_s, d_l],
                ebc_attenuation_slope_long_separation(mu_s, mu_a, d_s, d_l),
                "numpy",
            )

        return lambdify(
            [mu_s, mu_a, rho],
            ebc_attenuation_slope_short_separation(mu_s, mu_a, rho),
            "numpy",
        )
