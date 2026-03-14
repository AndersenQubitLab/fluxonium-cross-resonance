from __future__ import annotations

from math import pi
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


class Signal:
    def __init__(
            self,
            envelope: Callable[[np.floating|NDArray[np.floating]], ArrayLike],
            carrier_freq: float = 0.0,
            phase: float = 0.0,
    ) -> None:
        self.envelope = envelope
        self.carrier_freq = carrier_freq
        self.phase = phase

    def __call__(self, t: ArrayLike) -> np.complexfloating|NDArray[np.complexfloating]:
        t = np.asarray(t)
        carrier_points = np.exp(1j*(self.carrier_freq * t + self.phase))
        envelope_points = np.asarray(self.envelope(t))
        return np.real(envelope_points * carrier_points)


def planck_taper_signal(
        amplitude: float,
        total_duration: float,
        ramp_duration: float ,
        carrier_freq: float = 0.0,
        phase: float = 0.0,
        t0: float = 0.0,
) -> Signal:
    def envelope(t):
        t = t - t0

        # (Relic from when this code used JAX.)
        # See https://jax.readthedocs.io/en/latest/faq.html#gradients-contain-nan-where-using-where
        t_safe = np.where(
            (t != 0) & (t != ramp_duration) & (t != total_duration - ramp_duration) & (t != total_duration),
            t,
            0.0,
        )

        return amplitude * np.piecewise(
            t_safe,
            [
                (t > 0) & (t < ramp_duration),
                (t >= ramp_duration) & (t <= total_duration - ramp_duration),
                (t > total_duration - ramp_duration) & (t < total_duration),
            ],
            [
                ramp,
                1.0,
                lambda t: ramp(total_duration - t),
                0.0,
            ]
        )

    def ramp(t):
        # Prevent exponential from blowing up near zero
        t = np.clip(t, 0.01*ramp_duration, None)
        return (1 + np.exp(np.divide(ramp_duration, t) - np.divide(ramp_duration, ramp_duration - t)))**-1

    return Signal(
        envelope=envelope,
        carrier_freq=carrier_freq,
        phase=phase,
    )


def cosine_taper_signal(
        amplitude: float,
        total_duration: float,
        ramp_duration: float ,
        carrier_freq: float = 0.0,
        phase: float = 0.0,
        t0: float = 0.0,
) -> Signal:
    def envelope(t):
        t = t - t0

        return amplitude * np.where(
            t < 0,
            0.0,
            np.where(
                t >= total_duration,
                0.0,
                np.where(
                    t < ramp_duration,
                    0.5 * (1 - np.cos(np.divide(pi*t, ramp_duration))),
                    np.where(
                        t < total_duration - ramp_duration,
                        1.0,
                        0.5 * (1 - np.cos(np.divide(pi*(t - total_duration), ramp_duration))),
                    ),
                ),
            )
        )

    return Signal(
        envelope=envelope,
        carrier_freq=carrier_freq,
        phase=phase,
    )
