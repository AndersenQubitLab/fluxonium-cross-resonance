from collections.abc import Sequence
import copy
import functools
import operator

import numpy as np
from numpy.typing import ArrayLike


class FunctionComposition:
    def __init__(self, f, g) -> None:
        self.f = f
        self.g = g

    def __call__(self, x):
        return self.f(self.g(x))

    def derivative(self, nu: int = 1):
        if nu == 0:
            return self
        elif nu == 1:
            return FunctionProduct([
                FunctionComposition(self.f.derivative(1), self.g),
                self.g.derivative(1)
            ])
        elif nu == 2:
            return FunctionSum([
                FunctionProduct([
                    FunctionComposition(self.f.derivative(2), self.g),
                    FunctionProduct([self.g.derivative(1), self.g.derivative(1)]),
                ]),
                FunctionProduct([
                    FunctionComposition(self.f.derivative(1), self.g),
                    self.g.derivative(2),
                ])
            ])
        else:
            return self.derivative(1).derivative(nu - 1)


class FunctionSum:
    def __init__(
            self,
            args,
            coeffs: Sequence[int|float|complex]|None = None,
            constant: int|float|complex|None = None,
    ) -> None:
        if coeffs is not None and not (len(coeffs) == len(args)):
            raise ValueError(f"{len(coeffs)=} must equal {len(args)=}")

        self.args = args

        if coeffs is not None:
            self.coeffs = tuple(coeffs)
        else:
            self.coeffs = None

        self.constant = constant

    def __call__(self, x):
        if self.coeffs is None:
            value = sum(f(x) for f in self.args)
        else:
            value = sum(c*f(x) for c, f in zip(self.coeffs, self.args))

        if self.constant is not None:
            value = value + self.constant

        return value

    def derivative(self, nu: int):
        return FunctionSum([f.derivative(nu) for f in self.args], self.coeffs)


class FunctionProduct:
    def __init__(self, args) -> None:
        self.args = args

    def __call__(self, x):
        return functools.reduce(
            operator.mul,
            (f(x) for f in self.args),
        )

    def derivative(self, nu: int):
        if nu == 0:
            return self
        elif nu > 1:
            return self.derivative(1).derivative(nu - 1)

        return FunctionSum([
            FunctionProduct([
                f.derivative(1 if j == i else 0)
                for j, f in enumerate(self.args)
            ])
            for i in range(len(self.args))
        ])


class SampledFunction:
    def __init__(self, x: ArrayLike, *y: ArrayLike) -> None:
        self.x = np.asarray(x)
        self.y = tuple(np.asarray(yi) for yi in y)

    def __call__(self, x):
        # Linearly interpolate between sampled points
        return np.interp(x, self.x, self.y[0])

    def derivative(self, nu: int = 1):
        if nu < 0 or nu >= len(self.y):
            raise ValueError(nu)

        obj = copy.copy(self)
        obj.y = obj.y[nu:]

        return obj
