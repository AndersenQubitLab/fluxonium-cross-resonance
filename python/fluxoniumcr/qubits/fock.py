from __future__ import annotations

import numpy as np

from numpy.typing import NDArray

from .basis import Basis


class FockBasis(Basis):
    def _create_operator(self, name: str) -> NDArray:
        if name == 'create':
            return self.get_operator('destroy', stretch=False).swapaxes(-2, -1)
        elif name == 'destroy':
            return self._create_destroy_operator()
        elif name == 'number':
            return self._create_number_operator()
        else:
            return super()._create_operator(name)

    def _create_destroy_operator(self) -> NDArray:
        op = np.diag(np.arange(1, self.dim) ** 0.5, k=1)
        return op

    def _create_number_operator(self) -> NDArray:
        op = np.diag(np.arange(self.dim, dtype=float))
        return op
