from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from injector import Inject, Injector, singleton
import numpy as np
from numpy.typing import ArrayLike, NDArray

from fluxoniumcr.magnus import sesolve_magnusgl6

from .operator_resolver import OperatorResolver
from .signals import Signal
from .solve_methods import (
    MagnusGL6Method,
)


__all__ = (
    'Solver',
    'SolverResult',
)


class SolverResult(Protocol):
    t: NDArray|Sequence
    y: NDArray|Sequence[NDArray]


@singleton
class Solver:
    def __init__(self, injector: Inject[Injector]) -> None:
        self._injector = injector
        self._operator_resolver = injector.get(OperatorResolver)

    def solve(
            self,
            operators: Sequence[ArrayLike|str],
            signals: Sequence[Signal],
            tspan: tuple[float, float],
            method: MagnusGL6Method|None = None,
    ) -> SolverResult:
        if method is None:
            method = self._injector.get(MagnusGL6Method)

        injector = self._injector
        op_resolver = self._operator_resolver

        operators_parsed = []
        for op in operators:
            if isinstance(op, str):
                op = op_resolver.resolve(op)
            operators_parsed.append(op)
        H0 = op_resolver.H0

        solve_kwargs = dict(
            H0=H0,
            operators=operators_parsed,
            signals=tuple(signals),
            tspan=tspan,
            method=method,
        )

        if isinstance(method, MagnusGL6Method):
            return injector.call_with_injection(
                solve_unitary_magnusgl6,
                kwargs=solve_kwargs,
            )
        else:
            raise TypeError(f"Unknown method {method}")


def solve_unitary_magnusgl6(
        H0: ArrayLike,
        operators: Sequence[ArrayLike],
        signals: Sequence[Signal],
        tspan: tuple[float, float],
        *,
        method: MagnusGL6Method,
) -> SolverResult:
    H0 = np.asarray(H0, dtype=complex)
    operators = [np.asarray(op, dtype=complex) for op in operators]
    dt = method.dt

    n_steps = (tspan[1] - tspan[0])/dt
    if abs(n_steps - round(n_steps)) > 1e-5:
        raise ValueError(f"{(tspan[1]-tspan[0])=} is not a multiple of {dt=}")
    n_steps = round(n_steps)

    result = sesolve_magnusgl6(
        [
            H0,
            *([op, sig] for op, sig in zip(operators, signals)),
        ],
        psi0=np.identity(np.shape(H0)[0]),
        tlist=np.linspace(*tspan, n_steps+1)
    )

    return result
