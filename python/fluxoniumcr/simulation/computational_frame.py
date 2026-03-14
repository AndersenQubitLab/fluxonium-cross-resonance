from __future__ import annotations

from collections.abc import Mapping, Sequence
import itertools
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


class ComputationalFrame:
    def __init__(
            self,
            computational_indices: ArrayLike,
            qubit_order: Sequence,
            qubit_freqs: Mapping[Any, float],
    ) -> None:
        self._computational_indices = np.asarray(computational_indices, dtype=int)
        self._qubit_order = qubit_order
        self._qubit_freqs = dict(qubit_freqs)
        self._num_qubits = len(qubit_order)

    def transform(
            self,
            U: ArrayLike,
            tspan: ArrayLike,
            qubit_freqs: Mapping[Any, float]|None = None,
    ) -> NDArray:
        U = np.asarray(U)
        tspan = np.asarray(tspan)
        computational_indices = self._computational_indices
        qubit_order = self._qubit_order
        num_qubits = self._num_qubits

        if qubit_freqs is None:
            qubit_freqs = self._qubit_freqs
        else:
            qubit_freqs = self._qubit_freqs | dict(qubit_freqs)

        generator_diag = ([
            sum(
                z*qubit_freqs[qubit_order[i]]
                for i, z in enumerate(idx)
            )
            for idx in itertools.product(range(2), repeat=num_qubits)
        ])

        F0, F1 = (
                np.stack(
                [
                    np.exp(1j*(sign*t)*freq)
                    for freq in generator_diag
                ],
                axis=-1,
            )
            for sign, t in zip([-1, +1], tspan)
        )

        M = U[(..., *np.ix_(computational_indices, computational_indices))]
        M = M * F0[..., None, :]*F1[..., :, None]

        return M
