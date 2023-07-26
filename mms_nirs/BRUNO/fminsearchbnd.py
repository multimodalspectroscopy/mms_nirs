from enum import Enum, auto

import numpy as np
from scipy.optimize import OptimizeResult, minimize

# ADapted from the Matlab fminsearchbnd function
# https://uk.mathworks.com/matlabcentral/fileexchange/8277-fminsearchbnd-fminsearchcon


class BoundClass(Enum):
    UNCONSTRAINED = auto()
    LB = auto()
    UB = auto()
    BOTH = auto()
    FIXED_VAR = auto()


def get_bound_class(lb: float, ub: float) -> BoundClass:
    finite_bounds: tuple[bool, bool] = (
        bool(np.isfinite(lb)),
        bool(np.isfinite(ub)),
    )
    match finite_bounds:
        case (False, False):
            return BoundClass.UNCONSTRAINED
        case (True, False):
            return BoundClass.LB
        case (False, True):
            return BoundClass.UB
        case _:
            if lb == ub:
                return BoundClass.FIXED_VAR
            else:
                return BoundClass.BOTH


def xtransform_to_unconstrained(x0, params):
    # transform starting values into their unconstrained
    # surrogates. Check for infeasible starting guesses.
    x0u = []
    k = 0
    LB = params["LB"]
    UB = params["UB"]
    n = len(x0)
    for i in range(n):
        match params["BoundClass"][i]:
            case BoundClass.UNCONSTRAINED:
                x0u.append(x0[i])
                k += 1
            case BoundClass.LB:
                if x0[i] <= LB[i]:
                    x0u.append(0)
                else:
                    x0u.append(np.sqrt(x0[i] - LB[i]))
                k += 1
            case BoundClass.UB:
                if x0[i] >= UB[i]:
                    x0u.append(0)
                else:
                    x0u.append(np.sqrt(UB[i] - x0[i]))
                k += 1
            case BoundClass.BOTH:
                if x0[i] <= LB[i]:
                    x0u.append(-np.pi / 2)
                elif x0[i] >= UB[i]:
                    x0u.append(np.pi / 2)
                else:
                    temp = 2 * (x0[i] - LB[i]) / (UB[i] - LB[i]) - 1
                    # shift by 2*pi to avoid problems at zero in fminsearch
                    # otherwise, the initial simplex is vanishingly small
                    x0u.append(2 * np.pi + np.arcsin(np.clip(temp, -1, 1)))
                k += 1
            case BoundClass.FIXED_VAR:
                # fixed variable. drop it before fminsearch sees it.
                # k is not incremented for this variable.
                pass
    return np.array(x0u)


def xtransform_to_constrained(x0u, params):
    xtrans = []
    k = 0
    for i in range(params["n"]):
        match params["BoundClass"][i]:
            case BoundClass.UNCONSTRAINED:
                xtrans.append(x0u[k])
                k += 1
            case BoundClass.LB:
                xtrans.append(params["LB"][i] + x0u[k] ** 2)
                k += 1
            case BoundClass.UB:
                xtrans.append(params["UB"][i] - x0u[k] ** 2)
                k += 1
            case BoundClass.BOTH:
                temp = ((np.sin(x0u[k]) + 1) / 2) * (
                    params["UB"][i] - params["LB"][i]
                ) + params["LB"][i]

                xtrans.append(
                    np.maximum(
                        params["LB"][i], np.minimum(params["UB"][i], temp)
                    )
                )
                k += 1
            case BoundClass.FIXED_VAR:
                xtrans.append(params["LB"][i])
    return np.array(xtrans)


Nfeval = 1


def fminsearchbnd(
    fun, x0, LB=None, UB=None, options=None, func_args=[], *args, **kwargs
):
    def intrafun(x, params):
        xtrans = xtransform_to_constrained(x, params).reshape(params["xsize"])
        return fun(xtrans, *params["args"])

    xsize = np.atleast_1d(np.asarray(x0).shape)
    x0 = np.asarray(x0).ravel()
    n = len(x0)

    if LB is None or len(LB) == 0:
        LB = -np.inf * np.ones(n)
    else:
        LB = np.asarray(LB).ravel()

    if UB is None or len(UB) == 0:
        UB = np.inf * np.ones(n)
    else:
        UB = np.asarray(UB).ravel()

    if n != len(LB) or n != len(UB):
        raise ValueError("x0 is incompatible in size with either LB or UB.")

    if options is None:
        options = {"disp": False}

    params = {
        "args": func_args,
        "LB": LB,
        "UB": UB,
        "fun": fun,
        "n": n,
        "xsize": xsize,
        "OutputFcn": None,
    }

    params["BoundClass"] = [get_bound_class(LB[i], UB[i]) for i in range(n)]

    x0_unconstrained = xtransform_to_unconstrained(x0, params)

    if len(x0_unconstrained) == 0:
        # All variables were fixed. quit immediately
        result = OptimizeResult()
        result["x"] = x0
        result["success"] = False
        result["fun"] = fun(x0, *args)

        return result

    def callbackFn(Xi):
        f = intrafun(Xi, params)
        global Nfeval
        print(
            "{0:4d}   {1: 3.6e}   {2: 3.6e}   {3: 3.6e}   {4: 3.6e}   {5: 3.6e}   {6: 3.6e}".format(
                Nfeval, Xi[0], Xi[1], Xi[2], Xi[3], Xi[4], f
            )
        )
        Nfeval += 1

    callback = None

    if options["disp"]:
        callback = callbackFn

    result = minimize(
        intrafun,
        x0_unconstrained,
        args=(params,),
        **kwargs,
        options=options,
        method="Nelder-Mead",
        callback=callback,
    )

    x_unconstrained = result.x
    x = xtransform_to_constrained(x_unconstrained, params).reshape(xsize)

    transformed_result = result.copy()
    transformed_result["x"] = x
    return transformed_result
