from __future__ import annotations

from collections.abc import Callable
import copy
import dataclasses
import itertools
import math
from typing import Literal, TypedDict, overload

from injector import Inject
import numpy as np
from numpy.typing import NDArray

from .computational_frame import ComputationalFrame
from .floquet_solution import FloquetSolution, PeriodicSolution
from .signals import Signal
from .solve import Solver, SolverResult
from .solve_methods import MagnusGL6Method


class SmoothSquareParameters(TypedDict):
    total_duration: float
    ramp_duration: float
    amplitude: float
    carrier_freq: float


@dataclasses.dataclass
class CNOTParameters:
    pulse_parameters: SmoothSquareParameters
    pulse_factory: Callable[..., Signal]
    control_phase_correction: float = 0.0
    target_rotation_correction: float = 0.0
    drive_operator: str = 'q0.charge'

    def copy(self) -> CNOTParameters:
        return copy.deepcopy(self)


class CNOTSolver:
    def __init__(
            self,
            *,
            solver: Inject[Solver],
            magnus_method: Inject[MagnusGL6Method],
            computational_frame: Inject[ComputationalFrame],
    ) -> None:
        self._solver = solver
        self._computational_frame = computational_frame
        self._magnus_method = magnus_method

    def solve(
            self,
            gate_parameters: CNOTParameters,
            return_half: bool = False,
            return_unitary: bool = False,
            n_floquet_steps: int = 256,
    ) -> NDArray[np.complexfloating]:
        pulse_parameters = gate_parameters.pulse_parameters
        carrier_freq = pulse_parameters['carrier_freq']
        ramp_duration = pulse_parameters['ramp_duration']
        total_duration = pulse_parameters['total_duration']

        plateau_duration = total_duration - 2*ramp_duration
        ramp_phase = -math.pi/2 + plateau_duration/2 * carrier_freq
        floquet_solution = self.solve_floquet(gate_parameters, n_steps=n_floquet_steps)

        U_ramp = self.solve_ramp(gate_parameters, phase=ramp_phase)
        U_plateau = floquet_solution.dense(plateau_duration/2)
        U_half = U_ramp @ U_plateau

        if return_half:
            return U_half

        U = U_half @ U_half.swapaxes(-2, -1)

        if return_unitary:
            return U

        M = self._computational_frame.transform(
            U,
            tspan=(-total_duration/2, total_duration/2),
            qubit_freqs=dict(Q1=carrier_freq),
        )

        return M

    @overload
    def solve_ramp(
            self,
            gate_parameters: CNOTParameters,
            phase: float = -math.pi/2,
            *,
            return_full: Literal[False] = False,
    ) -> NDArray[np.complexfloating]:
        ...
    @overload
    def solve_ramp(
            self,
            gate_parameters: CNOTParameters,
            phase: float = -math.pi/2,
            *,
            return_full: Literal[True],
    ) -> SolverResult:
        ...
    def solve_ramp(
            self,
            gate_parameters: CNOTParameters,
            phase: float = -math.pi/2,
            return_full: bool = False,
    ):
        ramp_duration = gate_parameters.pulse_parameters['ramp_duration']
        new_gate_params = gate_parameters.copy()
        new_gate_params.pulse_parameters['total_duration'] = 2*ramp_duration

        # Round dt so that ramp_duration/dt is an integer.
        default_method = self._magnus_method
        method = default_method.replace(
            dt=ramp_duration/math.ceil(ramp_duration/default_method.dt)
        )

        drive_signal = gate_parameters.pulse_factory(
            **new_gate_params.pulse_parameters,
            phase=phase,
            t0=-ramp_duration,
        )

        result = self._solver.solve(
            operators=[new_gate_params.drive_operator],
            signals=[drive_signal],
            tspan=(0.0, ramp_duration),
            method=method,
        )

        if return_full:
            return result
        else:
            return np.asarray(result.y)[-1]

    def solve_floquet(
            self,
            gate_parameters: CNOTParameters,
            n_steps: int = 256,
    ) -> FloquetSolution:
        solver = self._solver

        pulse_parameters = gate_parameters.pulse_parameters
        ramp_duration = pulse_parameters['ramp_duration']
        carrier_freq = pulse_parameters['carrier_freq']
        carrier_period = 2*math.pi/carrier_freq

        drive_signal = gate_parameters.pulse_factory(
            **dict(pulse_parameters, total_duration=2*ramp_duration + carrier_period),
            phase=-math.pi/2,
            t0=-ramp_duration,
        )

        result = solver.solve(
            operators=[gate_parameters.drive_operator],
            signals=[drive_signal],
            tspan=(0.0, carrier_period),
            method=MagnusGL6Method(
                dt=carrier_period/n_steps
            ),
        )

        return FloquetSolution(result.t, result.y)

    def estimate_cnot_total_duration(
            self,
            gate_parameters: CNOTParameters,
    ) -> float:
        computational_frame = self._computational_frame
        pulse_parameters = gate_parameters.pulse_parameters

        ramp_duration = pulse_parameters['ramp_duration']
        carrier_freq = pulse_parameters['carrier_freq']
        carrier_period = 2*math.pi/carrier_freq

        # Unitary of one carrier period in the plateau.
        U_period = self.solve_floquet(gate_parameters).y[-1]
        # Unitary of the ramp down portion.
        U_ramp = self.solve_ramp(gate_parameters)

        def rotation_error(repetitions: int) -> float:
            U = (
                U_ramp
                @ np.linalg.matrix_power(U_period, repetitions)
                @ U_ramp.T
            )

            total_duration = 2*ramp_duration + repetitions * carrier_period
            M = computational_frame.transform(
                U,
                tspan=(-total_duration/2, total_duration/2),
                qubit_freqs=dict(Q1=carrier_freq),
            )

            # See equations (44), (45) from: V. Tripathi et al. Phys. Rev. A 100, 012301 (2019).
            conditional_rotation = -np.angle(
                (M[0,0] + M[1,1] + M[0,1] + M[1,0])
                / (M[2,2] + M[3,3] + M[2,3] + M[3,2])
                * (M[2,2] + M[3,3] - M[2,3] - M[3,2])
                / (M[0,0] + M[1,1] - M[0,1] - M[1,0])
            )

            error = conditional_rotation - math.pi

            # Wrap rotation error into the principal domain [-pi, pi)
            error = (error + math.pi) % (2*math.pi) - math.pi

            return error.item()

        nopt = integer_secant_method(rotation_error, x0=0, x1=10, interpolate=True)
        total_duration = 2*ramp_duration + nopt*carrier_period

        return total_duration

    def create_duration_sweep(
            self,
            gate_parameters: CNOTParameters,
            n_steps: int = 256,
            n_phases: int = 32,
    ) -> CNOTDurationSweep:
        pulse_parameters = gate_parameters.pulse_parameters
        carrier_freq = pulse_parameters['carrier_freq']
        ramp_duration = pulse_parameters['ramp_duration']

        p = gate_parameters.copy()
        p.pulse_parameters['total_duration'] = 2*ramp_duration
        t_data = np.linspace(0, 2*math.pi, n_phases + 1)
        y_data = []
        for theta in t_data[:-1]:
            y = self.solve_ramp(p, theta)
            y_data.append(y)

        # Last phase is 2*pi, which is identical to the first phase of 0.
        y_data.append(y_data[0])

        ramp_solution = PeriodicSolution(t_data, y_data)
        floquet_solution = self.solve_floquet(gate_parameters, n_steps=n_steps)

        return CNOTDurationSweep(
            carrier_freq,
            ramp_duration,
            floquet_solution,
            ramp_solution,
            self._computational_frame,
        )

    def calculate_transition_probabilities(
            self,
            gate_parameters: CNOTParameters,
    ) -> np.ndarray:
        ramp_duration = gate_parameters.pulse_parameters['ramp_duration']
        p = gate_parameters.copy()
        p.pulse_parameters['total_duration'] = 2*ramp_duration

        U_ramp = self.solve_ramp(p)
        floquet_solution = self.solve_floquet(gate_parameters)
        evecs = floquet_solution.eigenstates

        rho1 = np.abs(evecs.T.conj() @ U_ramp.T)**2
        rho1 = np.expand_dims(rho1.T, axis=2)
        rho2 = (U_ramp @ evecs) @ (rho1 * np.expand_dims((U_ramp @ evecs).T.conj(), axis=0))
        prob = np.diagonal(rho2, axis1=-2, axis2=-1).real.T

        return prob


class CNOTDurationSweep:
    def __init__(
            self,
            carrier_freq: float,
            ramp_duration: float,
            floquet_solution: FloquetSolution,
            ramp_solution: PeriodicSolution,
            computational_frame: ComputationalFrame,
    ) -> None:
        self._carrier_freq = carrier_freq
        self._ramp_duration = ramp_duration
        self._floquet_solution = floquet_solution
        self._ramp_solution = ramp_solution
        self._computational_frame = computational_frame

    @staticmethod
    def _solve_ramp_problem(
            cnot_solver: CNOTSolver,
            gate_parameters: CNOTParameters,
            n_phases: int,
    ) -> PeriodicSolution:
        """Solve the ramp unitary for all carrier phases from 0 to 2π."""
        pulse_parameters = gate_parameters.pulse_parameters
        ramp_duration = pulse_parameters['ramp_duration']
        new_gate_parameters = gate_parameters.copy()
        new_gate_parameters.pulse_parameters['total_duration'] = 2*ramp_duration

        t_data = np.linspace(0, 2*math.pi, n_phases + 1)
        y_data = []

        for theta in t_data[:-1]:
            y = cnot_solver.solve_ramp(new_gate_parameters, theta)
            y_data.append(y)

        # Last phase is 2*pi, which is identical to the first phase of 0.
        y_data.append(y_data[0])

        return PeriodicSolution(t_data, y_data)

    def solve(
            self,
            total_duration: float,
            return_half: bool = False,
            return_unitary: bool = False,
    ) -> NDArray[np.complexfloating]:
        ramp_duration = self._ramp_duration
        plateau_duration = total_duration - 2*ramp_duration
        ramp_phase = -math.pi/2 + plateau_duration/2 * self._carrier_freq
        U_ramp = self._ramp_solution.dense(ramp_phase)
        U_plateau = self._floquet_solution.dense(plateau_duration/2)
        U_half = U_ramp @ U_plateau

        if return_half:
            return U_half

        U = U_half @ U_half.swapaxes(-2, -1)

        if return_unitary:
            return U

        M = self._computational_frame.transform(
            U,
            tspan=(-total_duration/2, total_duration/2),
            qubit_freqs=dict(Q1=self._carrier_freq),
        )

        return M


def integer_secant_method(
        f: Callable[[int], float],
        x0: int,
        x1: int,
        maxiter: int = 50,
        brute: bool = True,
        interpolate: bool = False,
) -> float|int:
    x = x1

    for i in itertools.count():
        if maxiter and i >= maxiter: break
        if x0 == x1: break
        f0 = f(x0)
        f1 = f(x1)
        x = (x0*f1 - x1*f0)/(f1 - f0)
        x0 = x1
        x1 = round(x)

    if x0 != x1 and brute:
        xs = np.arange(min(x0, x1), max(x0, x1)+1)
        fs = np.array([f(x) for x in xs])
        return xs[np.argmin(np.abs(fs))]

    if interpolate:
        return x

    xopt = x1
    fopt = f(xopt)
    foptl = f(xopt - 1)
    foptr = f(xopt + 1)

    return (xopt, xopt - 1, xopt + 1)[np.argmin(np.abs([fopt, foptl, foptr]))]
