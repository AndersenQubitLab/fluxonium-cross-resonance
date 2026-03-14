import functools
from collections.abc import Mapping, Sequence

import math
import numpy as np
from numpy.typing import ArrayLike, NDArray
import scipy.optimize

from fluxoniumcr.qubits.basis import Basis


class ProductBasis:
    def __init__(
            self,
            subsystems: Sequence[Basis],
    ) -> None:
        self.subsystems = tuple(subsystems)
        self._operators: dict[tuple[str, ...], NDArray] = {}

        dims = tuple(s.dim for s in subsystems)
        strides: NDArray[np.integer] = np.append(np.cumprod(dims[:0:-1])[::-1], 1)

        self.dims = dims
        self._strides = strides

        # TODO: support aux_shape

    def get_operator(self, names: Mapping[int, str]|Sequence[str]) -> NDArray:
        if isinstance(names, Mapping):
            names_tuple = tuple(
                names.get(i, 'identity')
                for i in range(len(self.subsystems))
            )
        else:
            names_tuple = tuple(names)

        op = self._operators.get(names_tuple)
        if op is not None:
            return op

        op = self._create_operator(names_tuple)
        op.flags.writeable = False
        self._operators[names_tuple] = op
        return op

    def flat_index(self, indices: ArrayLike) -> NDArray:
        indices = np.asarray(indices)
        index = np.einsum('...i,i', indices, self._strides)
        return index

    def multi_index(self, index: ArrayLike) -> NDArray:
        index = np.asarray(index)
        indices = np.empty((*index.shape, len(self.subsystems)), dtype=int)

        rem = index
        for i, stride in enumerate(self._strides):
            num, rem = np.divmod(rem, stride)
            indices[..., i] = num

        indices = np.array(indices)
        return indices

    def _create_operator(self, names: Sequence[str]) -> NDArray:
        operators = [
            basis.get_operator(name)
            for basis, name in zip(self.subsystems, names)
        ]
        return functools.reduce(np.kron, operators)

    @property
    def dim(self) -> int:
        return math.prod(self.dims)


class DressedProductBasis:
    def __init__(
            self,
            hamiltonian: NDArray,
            bare_basis: ProductBasis,
            truncated_dims: Sequence[int],
    ) -> None:
        self.hamiltonian = hamiltonian.copy()
        self.hamiltonian.flags.writeable = False
        self.bare_basis = bare_basis
        self.truncated_dims = tuple(truncated_dims)

        self._strides = np.append(np.cumprod(truncated_dims[:0:-1])[::-1], 1)

        self._eigensystem: tuple[NDArray, NDArray]|None = None
        self._operators: dict[tuple[str, ...], NDArray] = {}

    def _solve_eigensystem(self) -> tuple[NDArray, NDArray]:
        evals, evecs = np.linalg.eigh(self.hamiltonian)

        keep_multi_indices = np.indices(self.truncated_dims)
        keep_multi_indices = keep_multi_indices.reshape(len(keep_multi_indices), -1).T
        keep_flat_indices = self.bare_basis.flat_index(keep_multi_indices)

        aux_shape = evals.shape[:-1]
        evals = evals.reshape(-1, evals.shape[-1])
        evecs = evecs.reshape(-1, *evecs.shape[-2:])
        new_evals = np.empty_like(
            evals,
            shape=(
                evals.shape[0],
                len(keep_flat_indices),
            ),
        )
        new_evecs = np.empty_like(
            evecs,
            shape=(
                evecs.shape[0],
                evecs.shape[-2],
                new_evals.shape[-1]
            ),
        )

        for i in range(evals.shape[0]):
            _, assign = scipy.optimize.linear_sum_assignment(
                abs(evecs[i, keep_flat_indices]),
                maximize=True,
            )

            new_evals[i] = evals[i, assign]
            new_evecs[i] = evecs[i][:, assign]

            phase_factor = np.exp(1j*np.angle(evecs[i][keep_flat_indices, assign]))
            new_evecs[i] /= phase_factor[None, :]

        new_evals = new_evals.reshape(*aux_shape, new_evals.shape[-1])
        new_evecs = new_evecs.reshape(*aux_shape, *new_evecs.shape[-2:])

        return new_evals, new_evecs

    def flat_index(self, indices: ArrayLike) -> int|NDArray:
        indices = np.asarray(indices)
        index = np.einsum('...i,i', indices, self._strides)
        return index

    def multi_index(self, index: ArrayLike) -> NDArray:
        index = np.asarray(index)
        indices = np.empty((*index.shape, len(self.subsystems)), dtype=int)

        rem = index
        for i, stride in enumerate(self._strides):
            num, rem = np.divmod(rem, stride)
            indices[..., i] = num

        indices = np.array(indices)
        return indices

    @property
    def subsystems(self) -> Sequence[Basis]:
        return self.bare_basis.subsystems

    def get_operator(self, names: Mapping[int, str]|Sequence[str]) -> NDArray:
        if isinstance(names,Mapping):
            names_tuple = tuple(
                names.get(i, 'identity')
                for i in range(len(self.subsystems))
            )
        else:
            names_tuple = tuple(names)

        op = self._operators.get(names_tuple)
        if op is not None:
            return op

        op = self._create_operator(names_tuple)
        op.flags.writeable = False
        self._operators[names_tuple] = op
        return op

    def _create_operator(self, names: Sequence[str]) -> NDArray:
        op = self.bare_basis.get_operator(names)
        evecs = self.eigenvectors
        op = evecs.swapaxes(-2, -1).conj() @ op @ evecs

        return op

    @property
    def eigenvalues(self) -> NDArray:
        return self.eigensystem[0]

    @property
    def eigenvectors(self) -> NDArray:
        """The eigenvectors in the bare basis."""
        return self.eigensystem[1]

    @property
    def eigensystem(self) -> tuple[NDArray, NDArray]:
        if self._eigensystem is None:
            esys = self._solve_eigensystem()
            self._eigensystem = esys
            self._eigensystem[0].flags.writeable = False
            self._eigensystem[1].flags.writeable = False
        return self._eigensystem

    @property
    def dim(self) -> int:
        return math.prod(self.truncated_dims)
