import copy

import numpy as np


class PlanckRampFunction:
    def __init__(
            self,
            ramp_duration: float,
            amplitude: float = 1.0,
    ) -> None:
        self.ramp_duration = ramp_duration
        self.amplitude = amplitude
        self._derivative_order = 0

    def __call__(self, t):
        x = t/self.ramp_duration
        if self._derivative_order == 0:
            y = planck_ramp_0(x)
        elif self._derivative_order == 1:
            y = planck_ramp_1(x)
        elif self._derivative_order == 2:
            y = planck_ramp_2(x)
        else:
            raise ValueError(self._derivative_order)
        return self.amplitude * y / self.ramp_duration ** self._derivative_order

    def derivative(self, nu: int = 1):
        obj = copy.copy(self)
        obj._derivative_order += nu

        if not (0 <= obj._derivative_order <= 2):
            raise ValueError(obj._derivative_order)

        return obj


def planck_ramp_0(x):
    x = np.float64(x)

    eps = 0.001409  # np.log2(np.e)/np.finfo(np.float64).maxexp

    def f(x: float):
        return (
            1 + np.exp(
                np.divide(1, x)
                - np.divide(1, 1 - x)
            )
        )**-1

    def f_small(x):
        return np.exp(np.divide(1, 1 - x) - np.divide(1, x))

    return np.piecewise(
        x,
        [x <= 0, (0 < x) & (x < eps), (eps <= x) & (x < 1)],
        [0, f_small, f, 1.0],
    )


def planck_ramp_1(x):
    x = np.float64(x)

    eps = 0.002772

    def f(x: float):
        return (1 + 2*x*(x-1)) * np.exp(1/(x-x**2))/(
            (np.exp(1/(1-x)) + np.exp(1/x))
            * x*(x-1)
        )**2

    return np.piecewise(
        x,
        [(eps <= x) & (x <= 1 - eps)],
        [f, 0.0],
    )


def planck_ramp_2(x):
    x = np.float64(x)

    eps = 0.004227

    def f(x):
        return np.exp(1/(x - x**2)) * (
            np.exp(1/(1-x)) * (-1 + 2*x - 4*x**3 + 6*x**4 - 4*x**5)
            + np.exp(1/x) * (1 - 6*x + 16*x**2 - 20*x**3 + 14*x**4 - 4*x**5)
        ) / (
            (np.exp(1/(1-x)) + np.exp(1/x))**3
            * x**4 * (x - 1)**4
        )

    return np.piecewise(
        x,
        [(eps <= x) & (x <= 1 - eps)],
        [f, 0.0],
    )
