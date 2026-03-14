from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Iterable, Sequence
import dataclasses
from math import pi
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse.csgraph import connected_components
import scipy.optimize
from sortedcontainers import SortedDict

from .magnus import sesolve_magnusgl6
from .utils import taylor_expand


@dataclasses.dataclass
class FloquetEigResult:
    angles: NDArray[np.floating]
    modes: NDArray[np.complexfloating]
    times: NDArray[np.floating]
    period: float

    freqs: NDArray[np.floating] = dataclasses.field(init=False)
    fft_freqs: NDArray[np.floating] = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.freqs = self.angles/self.period

        N = len(self.times)
        self.fft_freqs = 2*pi * np.fft.fftfreq(N, self.period/N)
        self.fft_freqs.flags.writeable = False

    def dress(self, op: NDArray) -> NDArray:
        modes = self.modes
        opI = modes.swapaxes(-2, -1).conj() @ op @ modes
        return opI

    def dress_fft(self, op: NDArray) -> NDArray:
        opI = self.dress(op)
        opI_fft = np.fft.fft(opI, axis=-3, norm='forward')
        return opI_fft


@dataclasses.dataclass
class FloquetEigDegResult(FloquetEigResult):
    eavgs: NDArray[np.floating]


@dataclasses.dataclass
class FloquetEigPerturbativeResult(FloquetEigResult):
    def dress(self, op: NDArray) -> NDArray:
        modes = self.modes
        opI = np.zeros((modes.shape[0], modes.shape[1], modes.shape[-1], modes.shape[-1]), dtype=complex)
        for n in range(opI.shape[0]):
            for k in range(n+1):
                opI[n] += modes[k].swapaxes(-2, -1).conj() @ op @ modes[n-k]
        return opI



class AdiabaticFloquetBasis:
    def __init__(
            self,
            H: list,
            T: float,
            N: int,
            **solver_kwargs,
    ) -> None:
        self._H = HamiltonianFunction.from_qutip_list(H)
        self._T = T
        self._N = N
        self._solver_kwargs = solver_kwargs

        self._fftfreq = 2*np.pi * np.fft.fftfreq(self._N, self._T/self._N)
        self._fftfreq.flags.writeable = False

        self._lookup = SortedDict()
        self._cache = SortedDict()

    def generate_lookup(
            self,
            params: Iterable[float],
            bare_modes: NDArray|None = None,
            bare_angles: NDArray|None = None,
            deg_tol: float|None = None,
    ) -> None:
        if len(self._lookup) > 0:
            self._lookup.clear()
            self._cache.clear()

        H0 = self._H.static_hamiltonian

        if bare_modes is None:
            #  _, bare_evecs = np.linalg.eigh(H0)
            # Assume H0 is diagonal.
            bare_evecs = np.identity(H0.shape[0])
            bare_angles = np.diag(H0) * self._T
            bare_modes = np.tile(bare_evecs, (self._N, 1, 1))
            if self._solver_kwargs.get('glide_reflection'):
                if self._solver_kwargs.get('parity_operator') is None:
                    offset_angles = np.zeros(bare_modes.shape[1])
                    offset_angles[1::2] = -2*pi
                    bare_modes = apply_phase_twist(bare_modes, offset_angles)
                    bare_angles += offset_angles
                else:
                    raise NotImplementedError(
                        "Calculating undriven modes for arbitrary parity operators is not implemented."
                    )

        if bare_angles is None:
            bare_angles = np.zeros(bare_modes.shape[0], dtype=float)

        override_solver_kwargs = {}
        if deg_tol is not None:
            override_solver_kwargs['deg_tol'] = deg_tol

        for p in sorted(params):
            self._query(
                p,
                bare_modes,
                bare_angles,
                for_lookup=True,
                override_solver_kwargs=override_solver_kwargs,
            )

    def quasienergies(self, p: float) -> NDArray:
        result = self.query(p)
        return result.angles/self._T

    def average_energies(self, p: float) -> NDArray:
        result = self.query(p)
        return result.eavgs

    def states(self, p: float) -> NDArray:
        result = self.query(p)
        return apply_phase_twist(result.modes, -result.angles)

    def modes(self, p: float) -> NDArray:
        result = self.query(p)
        return result.modes

    def query(self, p: float) -> FloquetEigDegResult:
        self._check_param_within_bounds(p)
        return self._query(p)

    def _check_param_within_bounds(self, p: float) -> None:
        if len(self._lookup) == 0:
            raise ValueError("call generate_lookup() first")

        # Keys are sorted because lookup is a SortedDict.
        lookup_params = self._lookup.keys()

        if p < lookup_params[0] or p > lookup_params[-1]:
            raise ValueError(
                f"argument {p} is outside of the precomputed domain ["
                f"{lookup_params[0]}, "
                f"{lookup_params[-1]}]"
            )

    def _query(
            self,
            p: float,
            bare_modes: NDArray|None = None,
            bare_angles: NDArray|None = None,
            for_lookup: bool = False,
            override_solver_kwargs: dict|None = None,
    ) -> FloquetEigDegResult:
        if not for_lookup and p in self._cache:
            return self._cache[p]

        result = self._solve_floquet_eigenproblem_for_parameter(
            p,
            override_solver_kwargs,
        )

        if len(self._lookup) == 0:
            if bare_modes is None or bare_angles is None:
                raise ValueError(
                    "`bare_modes` and `bare_angles` cannot be None if the lookup is empty."
                )
            nearest_modes = bare_modes
            nearest_angles = bare_angles
        else:
            nearest_modes = self._lookup[self._nearest_lookup(p)].modes
            nearest_angles = self._lookup[self._nearest_lookup(p)].angles

        new_angles, new_modes, permutation = connect_floquet_modes(
            result.angles,
            result.modes,
            nearest_modes,
            nearest_angles,
        )
        new_eavgs = result.eavgs[permutation]

        # Prevent accidentally overwriting calculated values.
        new_angles.flags.writeable = False
        new_modes.flags.writeable = False
        new_eavgs.flags.writeable = False

        new_result = dataclasses.replace(
            result,
            angles=new_angles,
            modes=new_modes,
            eavgs=new_eavgs,
        )

        if not for_lookup:
            self._cache[p] = new_result
        else:
            self._lookup[p] = new_result

        return new_result

    def query_perturbative(self, p: float, order: int = 1) -> FloquetEigPerturbativeResult:
        bare_result = self.query(p)

        times = bare_result.times
        coeffs_series = taylor_expand(
            lambda x: self._H.eval_coefficients(times[:, None], x[None]),
            p,
        )
        V_series = np.einsum('nkt,kij->ntij', coeffs_series[1:], self._H.operators)

        result = perturb_floquet_modes(
            bare_result,
            V_series,
            order=order,
        )

        return result

    def _solve_floquet_eigenproblem_for_parameter(
            self,
            p: float,
            override_solver_kwargs: dict|None = None,
    ) -> FloquetEigDegResult:
        H = self._H.to_qutip_list()
        Hp = [H[0]] + [
            [HC[0], lambda t, f=HC[1]: f(t, p)]
            for HC in H[1:]
        ]
        solver_kwargs = self._solver_kwargs.copy()
        if override_solver_kwargs is not None:
            solver_kwargs.update(override_solver_kwargs)
        return solve_floquet_eigenproblem(
            H=Hp,
            T=self._T,
            N=self._N,
            **solver_kwargs,
        )

    def _nearest_lookup(self, p: float) -> float:
        if len(self._lookup) == 0:
            raise ValueError("lookup is empty")

        # Keys are sorted because lookup is a SortedDict.
        lookup_params = self._lookup.keys()

        if p < lookup_params[0]:
            p_nearest = lookup_params[0]
        else:
            prev_index = self._lookup.bisect_right(p) - 1
            next_index = prev_index + 1
            p_prev = lookup_params[prev_index]
            if next_index >= len(lookup_params) or \
                    abs(p - p_prev) <= abs(p - (p_next := lookup_params[next_index])):
                p_nearest = p_prev
            else:
                p_nearest = p_next
        return p_nearest

    @property
    def fftfreq(self) -> NDArray:
        return self._fftfreq

    @property
    def domain(self) -> tuple[float, float]:
        return self._lookup.keys()[0], self._lookup.keys()[-1]

    @property
    def T(self) -> float:
        return self._T


class AbstractOperator(Protocol):
    @abstractmethod
    def __matmul__(self, other: NDArray) -> NDArray:
        """Return the result of left multiplying `other` by this operator.
        The last two axes of `other` are the matrix rows and columns."""

    @property
    @abstractmethod
    def T(self) -> AbstractOperator:
        """Return the transpose of this operation."""


class NegateEveryOtherBasisOperator(AbstractOperator):
    def __matmul__(self, operand: NDArray) -> NDArray:
        result = np.array(operand)
        result[..., ::2, :] *= -1
        return result

    @property
    def T(self) -> NegateEveryOtherBasisOperator:
        # This is a symmetric operator.
        return self


def solve_floquet_eigenproblem(
        H: list,
        T: float,
        N: int,
        deg_tol: float = 0.0,
        time_reversal: bool = False,
        glide_reflection: bool = False,
        parity_operator: AbstractOperator = NegateEveryOtherBasisOperator(),
) -> FloquetEigDegResult:
    result_nondeg = solve_floquet_eigenproblem_nondegenerate(
        H,
        T,
        N,
        time_reversal=time_reversal,
        glide_reflection=glide_reflection,
        parity_operator=parity_operator,
    )

    fangles = result_nondeg.angles
    fmodes = result_nondeg.modes
    times = result_nondeg.times

    if not glide_reflection:
        angle_mod = 2*pi
    else:
        angle_mod = 4*pi

    # Distance matrix between Floquet angles.
    dist_matrix = abs(np.mod(fangles[:, None] - fangles + angle_mod/2, angle_mod) - angle_mod/2)

    # Boolean matrix such that entry (i, j) is true if modes i and j are degenerate.
    deg_matrix = dist_matrix <= T * deg_tol

    H_func = HamiltonianFunction.from_qutip_list(H)
    H_samples = H_func(times)

    n_components, labels = connected_components(deg_matrix)
    deg_indices = [np.flatnonzero(labels == i) for i in range(n_components)]

    fangles2_list: list[float] = []
    fmodes2_list: list[NDArray] = []
    eavgs_list: list[float] = []
    for idx in deg_indices:
        fangles_deg_unwrapped = fangles[idx]
        fmodes_deg_unwrapped = fmodes[:, :, idx]

        # Wrap quasienergies within the same ±pi domain.
        if len(idx) > 1:
            fangles_deg = fangles_deg_unwrapped - angle_mod * np.round(
                (fangles_deg_unwrapped - sorted(fangles_deg_unwrapped)[len(idx)//2])
                / angle_mod
            )
            fmodes_deg = apply_phase_twist(
                fmodes_deg_unwrapped,
                fangles_deg - fangles_deg_unwrapped,
            )
        else:
            fangles_deg = fangles_deg_unwrapped
            fmodes_deg = fmodes_deg_unwrapped

        Eavg = np.mean(
            fmodes_deg.swapaxes(-2, -1).conj()
            @ H_samples
            @ fmodes_deg,
            axis=0,
        )

        if len(idx) == 1:
            fangles2_list.extend(fangles_deg)
            fmodes2_list.append(fmodes_deg)
            eavgs_list.append(Eavg.item().real)
            continue

        # Disable couplings between states that are not degenerate.
        Eavg[~deg_matrix[np.ix_(idx, idx)]] = 0.0

        Eavg_evals, Eavg_evecs = np.linalg.eigh(Eavg)

        # If the Floquet modes have all real coefficients, make sure we try to preserve this symmetry.
        largest_coeffs = np.take_along_axis(
            Eavg_evecs,
            np.argmax(abs(Eavg_evecs), axis=0)[None],
            axis=0,
        ).squeeze()
        Eavg_evecs *= np.exp(-1j*np.angle(largest_coeffs))[None]

        # Re-aligned floquet modes
        fangles_deg = np.abs(Eavg_evecs.T)**2 @ fangles_deg
        fmodes_deg = fmodes_deg @ Eavg_evecs

        fangles2_list.extend(fangles_deg)
        fmodes2_list.append(fmodes_deg)
        eavgs_list.extend(Eavg_evals)

    fangles2 = np.array(fangles2_list)
    fmodes2 = np.concatenate(fmodes2_list, axis=2)
    eavgs = np.array(eavgs_list)

    # Sort modes in order of ascending average energy.
    idx = np.argsort(eavgs)
    fangles2 = fangles2[idx]
    fmodes2 = fmodes2[..., idx]
    eavgs = eavgs[idx]

    result = FloquetEigDegResult(
        angles=fangles2,
        modes=fmodes2,
        eavgs=eavgs,
        times=times,
        period=T,
    )

    return result


def solve_floquet_eigenproblem_nondegenerate(
        H: list,
        T: float,
        N: int,
        time_reversal: bool = False,
        glide_reflection: bool = False,
        parity_operator: AbstractOperator = NegateEveryOtherBasisOperator(),
) -> FloquetEigResult:
    id_operator = np.identity(H[0].shape[0], dtype=complex)

    tlist = np.linspace(0, T, N+1)

    if (time_reversal, glide_reflection) == (False, False):
        sol = sesolve_magnusgl6(
            H,
            id_operator,
            tlist,
        )
        U = sol.y[-1]
        evals, evecs0 = np.linalg.eig(U)
        fangles = -np.angle(evals)
        fstates = sol.y[:-1] @ evecs0
        fmodes = apply_phase_twist(fstates, fangles)
    elif (time_reversal, glide_reflection) == (True, False):
        if N%2 != 0:
            raise ValueError(f"{N=} must be a multiple of 2 if {time_reversal=}")
        sol = sesolve_magnusgl6(
            H,
            id_operator,
            tlist[:N//2 + 1],
        )
        Uhalf = sol.y[-1]
        U = Uhalf.T @ Uhalf
        evals, evecs0 = np.linalg.eig(U)
        fangles = -np.angle(evals)
        fstates_half = sol.y @ evecs0
        fmodes_half = apply_phase_twist(fstates_half, fangles, hermitian=True)
        fmodes_half2 = fmodes_half[-2:0:-1].conj()
        fmodes = np.concatenate([fmodes_half, fmodes_half2], axis=0)
    elif (time_reversal, glide_reflection) == (False, True):
        if N%2 != 0:
            raise ValueError(f"{N=} must be a multiple of 2 if {glide_reflection=}")
        sol = sesolve_magnusgl6(
            H,
            id_operator,
            tlist[:N//2 + 1],
        )
        Uhalf = sol.y[-1]
        U = Uhalf.copy()
        U = parity_operator @ U
        evals, evecs0 = np.linalg.eig(U)
        fangles_half = -np.angle(evals)
        fangles = 2 * fangles_half
        fstates_half = sol.y[:-1] @ evecs0
        fmodes_half = apply_phase_twist(fstates_half, fangles_half)
        fmodes_half2 = parity_operator @ fmodes_half
        fmodes = np.concatenate([fmodes_half, fmodes_half2], axis=0)
    elif (time_reversal, glide_reflection) == (True, True):
        if N%4 != 0:
            raise ValueError(f"{N=} must be a multiple of 4 if {time_reversal=} and {glide_reflection=}")
        sol = sesolve_magnusgl6(
            H,
            id_operator,
            tlist[:N//4 + 1],
        )
        Q = sol.y[-1]
        U = Q.copy()
        U = parity_operator @ U
        U = Q.T @ U
        evals, evecs0 = np.linalg.eig(U)
        fangles_half = -np.angle(evals)
        fangles = 2 * fangles_half
        fstates_quarter = sol.y @ evecs0
        fmodes_quarter = apply_phase_twist(fstates_quarter, fangles_half, hermitian=True)
        fmodes_quarter2 = parity_operator @ fmodes_quarter[-2:0:-1].conj()
        fmodes_half = np.concatenate([fmodes_quarter, fmodes_quarter2], axis=0)
        fmodes_half2 = parity_operator @ fmodes_half
        fmodes = np.concatenate([fmodes_half, fmodes_half2], axis=0)
    else:
        assert False, "unreachable"

    result = FloquetEigResult(
        angles=fangles,
        modes=fmodes,
        times=tlist[:-1],
        period=T,
    )

    return result


def connect_floquet_modes(
        angles: NDArray,
        modes: NDArray,
        old_modes: NDArray,
        old_angles: NDArray,
) -> tuple[NDArray, NDArray, NDArray]:
    overlap = old_modes.swapaxes(1, 2).conj() @ modes

    overlap_fft = np.fft.fft(overlap, norm='forward', axis=0)
    fft_index = np.fft.fftfreq(overlap.shape[0], 1/overlap.shape[0]).astype(int)

    weights_all = np.real(overlap_fft.conj() * overlap_fft)

    idx = np.argmax(weights_all, axis=0)
    weights = np.take_along_axis(weights_all, idx[None], axis=0)[0]
    shifts = fft_index[idx]

    _, assign = scipy.optimize.linear_sum_assignment(
        weights,
        maximize=True,
    )

    shifts_assign = np.take_along_axis(shifts, assign[:, None], axis=1).squeeze()
    overlap_assign = np.take_along_axis(
        np.take_along_axis(overlap_fft, idx[None], axis=0).squeeze(),
        assign[:, None],
        axis=1,
    ).squeeze()

    phase_factor = np.sign(np.real(overlap_assign))

    new_angles = angles[assign] - shifts_assign * 2*pi
    new_modes = phase_factor * modes[:, :, assign]

    offset_angles = np.round((old_angles - new_angles)/(2*pi)) * 2*pi
    new_angles += offset_angles

    # Apply quasienergy shift to new modes.
    new_modes = apply_phase_twist(new_modes, -shifts_assign * 2*pi + offset_angles)

    return new_angles, new_modes, assign


def apply_phase_twist(
        states: NDArray,
        angles: NDArray,
        hermitian: bool = False,
) -> NDArray:
    if not hermitian:
        tlist = np.linspace(0, 1, states.shape[0], endpoint=False)
    else:
        tlist = np.linspace(0, 0.5, states.shape[0], endpoint=True)

    result = np.exp(1j * angles[None, None, :] * tlist[:, None, None]) * states

    return result


def perturb_floquet_modes(
        result: FloquetEigResult,
        V: NDArray,
        order: int = 3,
) -> FloquetEigPerturbativeResult:
    if order < 0 or order > 3:
        raise ValueError(f"{order=} must be between 0 and 3 (inclusive).")

    bare_freqs = result.freqs
    bare_modes = result.modes

    angles_series = np.zeros(
        (order + 1, *bare_freqs.shape),
        dtype=bare_freqs.dtype
    )
    modes_series = np.zeros(
        (order + 1, *bare_modes.shape),
        dtype=bare_modes.dtype
    )
    angles_series[0] = result.angles
    modes_series[0] = result.modes

    def make_result() -> FloquetEigPerturbativeResult:
        return FloquetEigPerturbativeResult(
            angles=angles_series,
            modes=modes_series,
            times=result.times,
            period=result.period,
        )

    if order == 0:
        return make_result()

    N = bare_modes.shape[0]
    dim = bare_modes.shape[1]

    fftfreq = 2*pi * np.fft.fftfreq(N, result.period/N)

    energy_denominator = (
        (bare_freqs[:, None] - bare_freqs[None, :])[None, :, :]
        + fftfreq[:, None, None]
    )
    # Avoid division by zero errors.
    energy_denominator[(0, *np.diag_indices(dim))] = 1.0

    # V in the bare modes basis.
    Vint = (
        bare_modes.swapaxes(-2, -1).conj()
        @ V
        @ bare_modes
    )

    Z1 = Vint[0]

    # Diagonal part of the perturbation
    Y1 = np.mean(Z1[(..., *np.diag_indices(dim))], axis=0)
    # Off-diagonal part of the perturbation
    X1 = Z1 - np.diag(Y1)[None]

    X1_fft = np.fft.fft(X1, axis=0)
    S1_fft = X1_fft/energy_denominator
    S1 = np.fft.ifft(S1_fft, axis=0)

    angles_series[1] = result.period * Y1.real
    # Note, S1 is in the bare_modes basis, so we do not need to act it on anything.
    modes_series[1] = bare_modes @ -S1

    if order <= 1:
        return make_result()

    C_S1_Y1 = S1*Y1[None, None, :]  - Y1[None, :, None]*S1
    C_S1_X1 = S1@X1 - X1@S1
    S1_squared = S1@S1

    Z2 = (
        Vint[1]
        + C_S1_Y1 + C_S1_X1/2
    )
    Y2 = np.mean(Z2[(..., *np.diag_indices(dim))], axis=0)
    X2 = Z2 - np.diag(Y2)[None]

    X2_fft = np.fft.fft(X2, axis=0)
    S2_fft = X2_fft/energy_denominator
    S2 = np.fft.ifft(S2_fft, axis=0)

    angles_series[2] = result.period * Y2.real
    modes_series[2] = bare_modes @ (-S2 + S1_squared/2)

    if order <= 2:
        return make_result()

    R = Vint[1] + C_S1_X1/3 + C_S1_Y1/2 - X2/2

    Z3 = (
        Vint[2]
        + (S2*Y1[None, None, :]  - Y1[None, :, None]*S2)
        + (S2@X1 - X1@S2)/2
        + S1@R - R@S1
    )
    Y3 = np.mean(Z3[(..., *np.diag_indices(dim))], axis=0)
    X3 = Z3 - np.diag(Y3)[None]

    X3_fft = np.fft.fft(X3, axis=0)
    S3_fft = X3_fft/energy_denominator
    S3 = np.fft.ifft(S3_fft, axis=0)

    angles_series[3] = result.period * Y3.real
    modes_series[3] = bare_modes @ (
        - S3
        + S1@(S2/2 - S1_squared/6)
        + S2@S1/2
    )

    return make_result()


class HamiltonianFunction:
    def __init__(
            self,
            static_hamiltonian: NDArray|None,
            operators: Sequence[NDArray],
            coefficients: Sequence[Callable],
    ) -> None:
        if static_hamiltonian is None and len(operators) == 0:
            raise ValueError("operators cannot be empty if {static_hamiltonian=}")
        elif static_hamiltonian is None:
            first_op = operators[0]
            static_hamiltonian = np.zeros(first_op.shape)

        self.static_hamiltonian = static_hamiltonian
        self.operators = operators
        self.coefficients = coefficients

    def __call__(self, t: ArrayLike, *args, **kwargs) -> NDArray:
        t = np.asarray(t)
        result: NDArray = sum(  # type: ignore
            op * f(t, *args, **kwargs)[:, None, None]
            for op, f in zip(self.operators, self.coefficients)
        )
        result += self.static_hamiltonian
        return result

    def eval_coefficients(self, t: ArrayLike, *args, **kwargs) -> NDArray:
        result = np.stack(
            [
                f(t, *args, **kwargs)
                for f in self.coefficients
            ],
            axis=0,
        )
        return result

    def to_qutip_list(self) -> list:
        H_list: list = [self.static_hamiltonian]
        for op, f in zip(self.operators, self.coefficients):
            H_list.append([op, f])
        return H_list

    @classmethod
    def from_qutip_list(cls, H: list) -> HamiltonianFunction:
        static_hamiltonian = None
        operators: list[NDArray] = []
        coefficients: list[Callable] = []
        for x in H:
            if len(x) == 2 and callable(x[1]):
                op = np.array(x[0])
                op.flags.writeable = False
                operators.append(op)
                coefficients.append(x[1])
            else:
                if static_hamiltonian is None:
                    # Make a copy of x
                    static_hamiltonian = np.array(x)
                else:
                    static_hamiltonian += x
        return cls(
            static_hamiltonian=static_hamiltonian,
            operators=operators,
            coefficients=coefficients,
        )


def isclose(a: float, b: float, abs_tol: float, modulus=None) -> bool:
    if modulus is None:
        return abs(a - b) <= abs_tol
    else:
        return abs(np.mod(a - b + modulus/2, modulus) - modulus/2) <= abs_tol
