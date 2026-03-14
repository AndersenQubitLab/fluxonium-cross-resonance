from collections.abc import Sequence
import itertools

from injector import Module, provider, singleton
import scipy.linalg

from fluxoniumcr.qubits.product_basis import DressedProductBasis

from .computational_frame import ComputationalFrame
from .operator_resolver import OperatorResolver
from .solve_methods import MagnusGL6Method


class SimulationModule(Module):
    def __init__(
            self,
            *,
            basis: DressedProductBasis,
            names: Sequence[str],
            dt: float,
    ) -> None:
        self._basis = basis
        self._names = names
        self._dt = dt

    @provider
    def basis(self) -> DressedProductBasis:
        return self._basis

    @singleton
    @provider
    def magnusgl6_method(self) -> MagnusGL6Method:
        return MagnusGL6Method(dt=self._dt)

    @singleton
    @provider
    def computational_frame(self) -> ComputationalFrame:
        basis = self._basis
        num_qubits = len(basis.truncated_dims)

        computational_indices = []
        for prod_idx in itertools.product(range(2), repeat=num_qubits):
            flag = False
            for subsys, i in enumerate(prod_idx):
                if i >= basis.truncated_dims[subsys]:
                    flag = True
                    break
            if flag:
                break
            computational_indices.append(basis.flat_index(prod_idx))
        computational_evals = basis.eigenvalues[computational_indices]

        ising_coeffs = 2**-num_qubits * scipy.linalg.hadamard(2**num_qubits) @ computational_evals
        qubit_freqs = {
            self._names[i]:
            -2*ising_coeffs[2**(num_qubits-1 - i)] for i in range(num_qubits)
        }

        return ComputationalFrame(
            computational_indices,
            qubit_order=self._names,
            qubit_freqs=qubit_freqs
        )

    @singleton
    @provider
    def operator_resolver(self) -> OperatorResolver:
        return OperatorResolver(
            basis=self._basis,
            names=self._names
        )
