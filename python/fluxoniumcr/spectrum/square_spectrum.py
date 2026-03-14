from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
import scipy.special


class SquareSpectrum:
    def __init__(
            self,
            numerator: Callable,
            pole: float,
            bare_pole: float = np.nan,
    ) -> None:
        self.numerator = numerator
        self.pole = pole
        self.bare_pole = bare_pole

    def __call__(self, w):
        denominator = (np.real(w) - self.pole) ** 2 + np.imag(w) ** 2
        if not np.isnan(self.bare_pole):
            denominator *= (np.real(w) - self.bare_pole) ** 2 + np.imag(w) ** 2
        return self.numerator(np.real(w))/denominator


class DTFTInterpolator:
    def __init__(
            self,
            a: ArrayLike|Sequence[ArrayLike],
            d: float = 1.0,
            shift: float = 0.0,
    ) -> None:
        # `a` should not be modified after it is used to construct this object.
        self._a = np.asarray(a, dtype=complex)
        self._w = 2*np.pi * np.fft.fftfreq(len(a), d=d)
        self._d = d
        self._shift = shift

    @property
    def a(self) -> NDArray[np.complexfloating]:
        # Do not modify the return value
        return self._a

    def __call__(self, w: ArrayLike) -> NDArray[np.complexfloating]:
        a_data = self._a
        w_data = self._w

        w = np.asarray(w)
        input_shape = w.shape
        w = w.ravel()

        def kernel(x, n):
            return np.exp(-1j*(n-1)*x/2) * scipy.special.diric(x, n)

        # Convolve with a periodic sinc kernel to resample.
        a = np.einsum(
            "j...,ji->i...",
            a_data,
            kernel(
                x=self._d * (w - w_data[:, None] - self._shift),
                n=len(a_data),
            ),
        )
        a = a.reshape((*input_shape, *a.shape[1:]))

        return a


def calculate_square_spectrum(
        r0,
        r1,
        r2,
        f0,
        f1,
        dt,
):
    if np.isclose(r0[0], 0.0, atol=1e-9):
        Fv = calculate_velocity_chirplet_ft(
            r0, r1, f0, f1, dt
        )
        spectrum = SquareSpectrum(
            numerator=lambda w: 2*abs(Fv(w))**2,
            pole=f0[-1],
        )
    else:
        Fa = calculate_acceleration_chirplet_ft(
            r0, r1, r2, f0, f1, dt
        )
        spectrum = SquareSpectrum(
            numerator=lambda w: 2*abs(Fa(w))**2,
            pole=f0[-1],
            bare_pole=f0[0],
        )

    return spectrum


def calculate_velocity_chirplet_ft(r0, r1, f0, f1, dt) -> DTFTInterpolator:
    window = (
        r1
        - 1j*(f0[-1] - f0) * r0
    )
    phase = trapezoid(f0 - f0[0], f1, dt)
    chirplet = window * np.exp(1j*phase)

    Fv = np.fft.fft(chirplet, norm='backward') * dt

    return DTFTInterpolator(Fv, dt, shift=f0[0])


def calculate_acceleration_chirplet_ft(r0, r1, r2, f0, f1, dt) -> DTFTInterpolator:
    window = (
        r2
        - 1j*(f0[-1] + f0[0] - 2*f0) * r1
        + (1j*f1 + (f0 - f0[0]) * (f0[-1] - f0)) * r0
    )
    phase = trapezoid(f0 - f0[0], f1, dt)
    chirplet = window * np.exp(1j*phase)

    Fa = np.fft.fft(chirplet, norm='backward') * dt

    return DTFTInterpolator(Fa, dt, shift=f0[0])


def trapezoid(y, dy=None, d=1.0):
    cumsum_y = np.cumsum(y)
    int_y = np.empty_like(cumsum_y)
    int_y[0] = 0.0
    int_y[1:] = d/2*(cumsum_y[:-1] + cumsum_y[1:])
    if dy is not None:
        int_y[1:] += d**2/12 * (dy[0] - dy[1:])
    return int_y
