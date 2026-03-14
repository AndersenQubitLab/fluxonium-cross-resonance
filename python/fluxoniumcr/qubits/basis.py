from __future__ import annotations

import numpy as np

from numpy.typing import ArrayLike, NDArray
from abc import ABCMeta, abstractmethod
from typing import Mapping, Sequence


class UnknownOperatorError(KeyError):
    pass


class Basis(metaclass=ABCMeta):
    def __init__(
            self,
            *,
            parameters: Mapping[str, ArrayLike]|None = None,
            dim: int,
            aux_shape: tuple[int, ...] = (),
            **kwargs,
    ) -> None:
        parameters = self._build_parameters(parameters, **kwargs)

        aux_shape = np.broadcast_shapes(*(
            aux_shape,
            *(np.shape(v) for v in parameters.values()),
        ))
        expanded_params: dict[str, NDArray] = {}
        for k, v in parameters.items():
            old_shape = np.shape(v)
            if old_shape != ():
                new_shape = (1,) * (len(aux_shape) - len(old_shape)) + old_shape
            else:
                new_shape = ()

            expanded_params[k] = np.reshape(v, new_shape)

        self._parameters = expanded_params

        self._dim = dim
        self._aux_shape = aux_shape
        self._operators: dict[str, NDArray] = {}

    def _build_parameters(
            self,
            parameters: Mapping[str, ArrayLike]|None = None,
            **kwargs,
    ) -> Mapping[str, ArrayLike]:
        if parameters is None:
            parameters = kwargs
        else:
            parameters = dict(parameters, **kwargs)

        for k, v in self._get_default_parameters():
            parameters.setdefault(k, v)

        return parameters

    @classmethod
    def _get_default_parameters(cls) -> tuple[tuple[str, ArrayLike], ...]:
        return tuple()

    def get_operator(self, name: str, stretch: bool = True) -> NDArray:
        op = self._get_unstretched_operator(name)
        if stretch:
            op = np.broadcast_to(op, self.aux_shape + (self.dim, self.dim))

        return op  # type: ignore

    def _get_unstretched_operator(self, name: str) -> NDArray:
        op = self._operators.get(name)
        if op is not None:
            return op

        op = self._create_operator(name)
        if self.aux_shape != ():
            if len(op.shape) == 2:
                op = np.expand_dims(op, axis=tuple(range(len(self.aux_shape))))
            else:
                # Nothing to do here.
                pass
        op.flags.writeable = False
        self._operators[name] = op  # type: ignore

        return op  # type: ignore

    def get_operators(
            self,
            names: Sequence[str],
            stretch: bool = True,
    ) -> Sequence[NDArray]:
        return tuple(self.get_operator(name, stretch=stretch) for name in names)

    def get_parameter(self, name: str, stretch: bool = True) -> NDArray:
        value = self._parameters[name]
        if stretch:
            value = np.broadcast_to(value, self.aux_shape)
        return value

    def get_parameters(self, names: Sequence[str], stretch: bool = True) -> Sequence[NDArray]:
        return tuple(self.get_parameter(name, stretch=stretch) for name in names)

    def _create_operator(self, name: str) -> NDArray:
        if name == 'identity':
            return np.identity(self.dim)
        else:
            raise UnknownOperatorError(name)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def aux_shape(self) -> tuple[int, ...]:
        return self._aux_shape


class Eigenbasis(Basis):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._super_basis = self._create_super_basis()
        self._hamiltonian: NDArray|None = None
        self._is_hamiltonian_diagonal = False
        self._eigensystem: tuple[NDArray, NDArray]|None = None
        self._is_change_of_basis_trivial = False

    @abstractmethod
    def _create_super_basis(self) -> Basis:
        """Create and return the basis of the to-be-diagonalized Hamiltonian."""

    @abstractmethod
    def _create_hamiltonian(self) -> NDArray:
        """Create and return the to-be-diagonalized Hamiltonian."""

    def _transform_eigensystem(self, evals: NDArray, evecs: NDArray) -> tuple[NDArray, NDArray]:
        return evals, evecs

    def _solve_eigensystem(self) -> tuple[NDArray, NDArray]|None:
        """Diagonalize the Hamiltonian and return the eigenvalues and eigenvectors.

        Subclasses can override this method to return None if the Hamiltonian is already
        diagonal. In which case, the eigenvalues are extracted from the first `dim`
        diagonal elements of the Hamiltonian and the eigenvectors are the first `dim`
        columns of the super basis identity."""
        evals, evecs = np.linalg.eigh(self.hamiltonian)
        evals = evals[..., :self.dim]
        evecs = evecs[..., :self.dim]
        return evals, evecs

    def _create_operator(self, name: str) -> NDArray:
        try:
            if name == 'hamiltonian':
                evals = self.eigenvalues
                op = np.zeros(evals.shape[:-1] + (self.dim, self.dim))
                op[(..., *np.diag_indices(self.dim))] = evals
                return op
            else:
                return super()._create_operator(name)
        except UnknownOperatorError:
            op = self.super_basis.get_operator(name, stretch=False)
            if not self._is_change_of_basis_trivial:
                evecs = self.eigenvectors
                op = evecs.swapaxes(-2, -1).conj() @ op @ evecs
            else:
                op = op[..., :self.dim, :self.dim]
            return op

    @property
    def super_basis(self) -> Basis:
        return self._super_basis

    @property
    def hamiltonian(self) -> NDArray:
        """The Hamiltonian in the super basis."""
        if self._hamiltonian is None:
            ham = self._create_hamiltonian()
            ham.flags.writeable = False
            self._hamiltonian = ham
        return self._hamiltonian

    @property
    def eigenvalues(self) -> NDArray:
        return self.eigensystem[0]

    @property
    def eigenvectors(self) -> NDArray:
        """The eigenvectors in the super basis."""
        return self.eigensystem[1]

    @property
    def eigensystem(self) -> tuple[NDArray, NDArray]:
        if self._eigensystem is None:
            esys = self._solve_eigensystem()
            if esys is not None:
                esys = self._transform_eigensystem(*esys)
            else:
                # If None is returned, assume Hamiltonian is diagonal.
                evals = self.hamiltonian[(..., *np.diag_indices(self.dim))]
                evecs = self.super_basis.get_operator('identity', stretch=True)
                evecs = evecs[..., :, :self.dim]
                esys = evals, evecs
                self._is_change_of_basis_trivial = True
            self._eigensystem = esys  # type: ignore
            self._eigensystem[0].flags.writeable = False  # type: ignore
            self._eigensystem[1].flags.writeable = False  # type: ignore
        return self._eigensystem  # type: ignore
