from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
import scipy.special

from .basis import Eigenbasis
from .fock import FockBasis


class Oscillator(Eigenbasis):
    def _create_super_basis(self) -> HermiteGaussianBasis:
        EC, EL = self.get_parameters(['EC', 'EL'], stretch=False)
        std = (2*EC/EL)**0.25
        return HermiteGaussianBasis(std=std, dim=self.dim)

    def _create_hamiltonian(self) -> NDArray[np.floating]:
        EC, EL = self.get_parameters(['EC', 'EL'], stretch=False)
        N, I = self.super_basis.get_operators(['number', 'identity'], stretch=False)
        H = ((8*EL*EC)**0.5)[..., None, None] * (N + 0.5*I)
        return H

    def _solve_eigensystem(self) -> None:
        # Eigenvectors are just columns of the identity matrix.
        return None


class HermiteGaussianBasis(FockBasis):
    @classmethod
    def _get_default_parameters(cls) -> tuple[tuple[str, ArrayLike], ...]:
        return super()._get_default_parameters() + (
            ('std', 1.0),
        )

    def _create_operator(self, name: str) -> NDArray[np.complexfloating]|NDArray[np.floating]:
        if name == 'phi':
            return self._create_phi_operator()
        elif name == 'charge':
            return self._create_charge_operator()
        elif name == 'exp_iphi':
            return self._create_exp_iphi_operator()
        elif name == 'cos_phi':
            return self.get_operator('exp_iphi', stretch=False).real
        elif name == 'sin_phi':
            return self.get_operator('exp_iphi', stretch=False).imag
        else:
            return super()._create_operator(name)

    def _create_phi_operator(self) -> NDArray[np.complexfloating]:
        a, adg = self.get_operators(['destroy', 'create'], stretch=False)
        std = self.get_parameter('std', stretch=False)
        return std[..., None, None] * (a + adg)

    def _create_charge_operator(self) -> NDArray[np.complexfloating]:
        a, adg = self.get_operators(['destroy', 'create'], stretch=False)
        std = self.get_parameter('std', stretch=False)
        return -0.5j/std[..., None, None] * (a - adg)

    def _create_exp_iphi_operator(self) -> NDArray[np.complexfloating]:
        std = self.get_parameter('std', stretch=False)
        ii, jj = np.triu_indices(self.dim)
        op = np.zeros(std.shape + (self.dim, self.dim), dtype=np.complex128)
        op[..., ii, jj] = calculate_exp_iphi_element(std[..., None], ii, jj)
        op += np.triu(op, k=1).swapaxes(-2, -1)
        return op


def calculate_exp_iphi_element(std: ArrayLike, i: ArrayLike, j: ArrayLike) -> ArrayLike:
    std = np.asarray(std)
    i = np.asarray(i)
    j = np.asarray(j)
    k = j - i

    elm = (
        1j ** k
        * np.exp(
            - 0.5*(
                scipy.special.loggamma(i+k+1)
                - scipy.special.loggamma(i+1)
                + std**2
            )
            + k*np.log(std)
        )
        * scipy.special.eval_genlaguerre(i, k, std**2)
    )
    return elm
