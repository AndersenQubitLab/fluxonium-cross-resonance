from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .basis import Eigenbasis
from .oscillator import Oscillator


class Fluxonium(Eigenbasis):
    @classmethod
    def _get_default_parameters(cls) -> tuple[tuple[str, ArrayLike], ...]:
        return super()._get_default_parameters() + (
            ('flux', 0.5),
            ('cutoff', 64),
        )

    def _create_super_basis(self) -> Oscillator:
        EC, EL, cutoff = self.get_parameters(['EC', 'EL', 'cutoff'], stretch=False)
        if cutoff.size > 1:
            raise ValueError(f"Oscillator cutoff must be a scalar, got `{cutoff}`.")
        return Oscillator(EC=EC, EL=EL, dim=cutoff, aux_shape=self.aux_shape)

    def _create_hamiltonian(self) -> NDArray[np.floating]:
        EJ, flux = self.get_parameters(['EJ', 'flux'], stretch=False)

        (
            H_osc,
            cos_phi,
            sin_phi,
        ) = self.super_basis.get_operators(
            [
                'hamiltonian',
                'cos_phi',
                'sin_phi',
            ],
            stretch=False,
        )

        EJ = EJ[..., None, None]
        flux = flux[..., None, None]

        H = H_osc  - EJ * (
            cos_phi * np.cos(2*np.pi*flux)
            - sin_phi * np.sin(2*np.pi*flux)
        )

        return H

    def _transform_eigensystem(self, evals: NDArray, evecs: NDArray) -> tuple[NDArray, NDArray]:
        # Make sure <i|a|i+1> is positive.
        destroy_bare = self.super_basis.get_operator('destroy')
        destroy = evecs.swapaxes(-2, -1).conj() @ destroy_bare @ evecs
        phases = np.cumprod(np.sign(np.diagonal(destroy, offset=1, axis1=-2, axis2=-1)), axis=-1)
        new_evecs = evecs.copy()
        new_evecs[..., 1:] *= phases[..., None, :]
        return evals, new_evecs
