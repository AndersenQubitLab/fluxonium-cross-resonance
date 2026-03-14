from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from fluxoniumcr.qubits.product_basis import DressedProductBasis


class OperatorResolver:
    def __init__(
            self,
            basis: DressedProductBasis,
            names: Sequence[str],
    ) -> None:
        self._basis = basis
        self._names = names
        self._H0: NDArray|None = None

    def resolve(self, op_str: str) -> NDArray:
        id_str, op_name = op_str.split('.')
        op_dict = {}
        op_dict[self._names.index(id_str)] = op_name
        op = self._basis.get_operator(op_dict)
        return op

    @property
    def H0(self) -> NDArray:
        if self._H0 is None:
            self._H0 = np.diag(self._basis.eigenvalues)
        return self._H0
