from __future__ import annotations

from math import pi
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
import scipy.special


__all__ = (
    'FloquetSolution',
)


class FloquetSolution:
    def __init__(
            self,
            t: ArrayLike|Sequence[float],
            y: ArrayLike|Sequence[ArrayLike],
            _evecs: NDArray[np.complexfloating]|None = None,
            _expos: NDArray[np.complexfloating]|None = None,
            _micromotion: PeriodicSolution|None = None,
    ) -> None:
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=complex)

        # t and y should not be modified after it is used to construct this object.
        self._t = t
        self._y = y

        self._micromotion = _micromotion
        self._evecs = _evecs
        self._expos = _expos

    @property
    def eigenstates(self) -> NDArray[np.complexfloating]:
        if self._evecs is None:
            self._solve_floquet()
            assert self._evecs is not None
        # Do not modify this return value
        return self._evecs

    @property
    def exponents(self) -> NDArray[np.complexfloating]:
        if self._expos is None:
            self._solve_floquet()
            assert self._expos is not None
        # Do not modify this return value
        return self._expos

    @property
    def micromotion(self) -> PeriodicSolution:
        if self._micromotion is None:
            self._solve_floquet()
            assert self._micromotion is not None
        # Do not modify this return value
        return self._micromotion

    def _solve_floquet(self) -> None:
        t = self.t
        y = self.y
        evals, evecs = np.linalg.eig(y[-1])

        # Complex Floquet characteristic exponents
        expos = 1j*np.angle(evals)/t[-1]

        F = y[:-1] @ (
            (evecs[None] * (np.exp(-t[:-1, None] * expos))[:, None, :]) @ evecs.T.conj()[None]
        )
        micromotion = PeriodicSolution(t, np.concatenate((F, F[None, 0])))

        self._evecs = evecs
        self._expos = expos
        self._micromotion = micromotion

    @property
    def t(self) -> NDArray[np.floating]:
        # Do not modify this return value
        return self._t

    @property
    def y(self) -> NDArray[np.floating]:
        # Do not modify this return value
        return self._y

    def dense(self, t: ArrayLike) -> NDArray[np.complexfloating]:
        t = np.asarray(t)
        input_shape = t.shape
        t = t.ravel()

        y = self.micromotion.dense(t) @ (
            (self.eigenstates[None] * np.exp(t[:, None] * self.exponents[None])[:, None])
            @ self.eigenstates.T.conj()[None]
        )
        y = y.reshape((*input_shape, *y.shape[1:]))

        return y


class PeriodicSolution:
    def __init__(
            self,
            t: ArrayLike|Sequence[float],
            y: ArrayLike|Sequence[ArrayLike],
    ) -> None:
        """Periodic solution.

        `t` must be evenly spaced with `t[-1]` equal to the period of the solution.
        The first and last points of `y` are assumed to be equal on account of periodicity.
        Dense evaluation uses sinc interpolation."""

        # t and y should not be modified after it is used to construct this object.
        self._t = np.asarray(t, dtype=float)
        self._y = np.asarray(y, dtype=complex)

    @property
    def t(self) -> NDArray[np.complexfloating]:
        # Do not modify the return value
        return self._t

    @property
    def y(self) -> NDArray[np.complexfloating]:
        # Do not modify the return value
        return self._y

    def dense(self, t: ArrayLike) -> NDArray[np.complexfloating]:
        t_data = self._t
        y_data = self._y

        t = np.asarray(t)
        input_shape = t.shape
        t = t.ravel()

        T = t_data[-1]
        N = t_data.shape[0] - 1
        # Convolve with a periodic sinc kernel to resample calculated points.
        y = np.einsum(
            "j...,ji->i...",
            y_data[:-1],
            sinc_comb((t - t_data[:-1, None]) * N/T, N),
        )
        y = y.reshape((*input_shape, *y.shape[1:]))

        return y


def sinc_comb_old(x: ArrayLike, n: int) -> np.floating|NDArray[np.floating]:
    r"""Return :math:`\sum^∞_{k=-∞} \frac{\sin(π(x - k*n))}{π(x - k*n)}`."""
    x = np.asarray(x, dtype=float)

    cosine_amp = (1 - np.fmod(n, 2))/n
    numerator = np.sin(pi*x * (1 - cosine_amp))
    denominator = n * np.sin(pi*x/n)

    epsilon = 1e-9
    denominator_safe = np.where(
        np.abs(denominator) > epsilon,
        denominator,
        # Resolve singularity with L'Hôpital's rule
        pi * np.cos(pi*x/n),
    )
    numerator_safe = np.where(
        np.abs(denominator) > epsilon,
        numerator,
        pi * (1 - cosine_amp) * np.cos(pi*x * (1 - cosine_amp)),
    )

    return cosine_amp*np.cos(pi*x) + numerator_safe/denominator_safe


def sinc_comb(x: ArrayLike, n: int) -> np.floating|NDArray[np.floating]:
    r"""Return :math:`\sum^∞_{k=-∞} \frac{\sin(π(x - k*n))}{π(x - k*n)}`."""
    x = np.asarray(x, dtype=float)

    # Equivalent to `np.sin(np.pi*x)/(n*np.sin(np.pi*x/n))` but without singularities.
    diric = scipy.special.diric(x * 2*np.pi/n, n)

    if n%2:
        return diric
    else:
        return diric * np.cos(pi*x/n)
