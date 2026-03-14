import numpy as np
from numpy.typing import ArrayLike, NDArray
import scipy.optimize

from .constants import DIELECTRIC_LOSS_TANGENT
from .floquet import AdiabaticFloquetBasis, FloquetEigResult
from .optimize import find_root, get_monotonic_increasing_intervals
from .qubits.basis import Basis
from .qubits.fluxonium import Fluxonium


def create_driven_fluxonium(
        fx: Fluxonium,
        drive_freq: float,
        phase_gauge: bool = True,
        **kwargs,
) -> AdiabaticFloquetBasis:
    H0_op, phi_op, n_op = fx.get_operators(['hamiltonian', 'phi', 'charge'])

    kwargs.setdefault('N', 256)
    kwargs.setdefault('deg_tol', 1e-9 * 2*np.pi)
    kwargs.setdefault('time_reversal', True)
    kwargs.setdefault('glide_reflection', True)

    EC = fx.get_parameter('EC').item()
    H_list: list = [H0_op]

    if phase_gauge:
        H_list.append([
            phi_op,
            lambda t, amp: -amp*drive_freq/(8*EC) * np.cos(drive_freq*t)
        ])
    else:
        H_list.append([
            n_op,
            lambda t, amp: amp * np.sin(drive_freq*t)
        ])

    floquet_basis = AdiabaticFloquetBasis(
        H_list,
        T=2*np.pi/drive_freq,
        **kwargs,
    )

    return floquet_basis


def calculate_critical_amplitude(
        qubit: Basis,
        floquet_basis: AdiabaticFloquetBasis,
        step_size: float,
        xtol: float,
        x0: float|None = None,
        x1: float|None = None,
        minimum_energy_gap: float = 10e-3 * 2*np.pi,
):
    if x0 is None:
        x0 = floquet_basis.domain[0]
    if x1 is None:
        x1 = floquet_basis.domain[1]

    def func(amp):
        result = floquet_basis.query_perturbative(amp, order=1)
        p0, p1, y = map(np.polynomial.Polynomial,
            calculate_polarization_and_error(qubit, result)
        )

        objective = (p1 - p0)*reciprocal(y)

        if len(objective.coef) <= 1:
            #  The objective just so happens to have an exactly zero tangent.
            return 0.0

        obj_grad = objective.coef[1]

        # Take gradient of absolute value of objective.
        if p0.coef[0] > p1.coef[0]:
            obj_grad *= -1

        return obj_grad

    xopt, is_zero = find_root(func, x0, x1, step_size=step_size, xtol=xtol)
    if xopt is not None and abs(func(xopt)) > 10*xtol/DIELECTRIC_LOSS_TANGENT:
        # XXX: Hacky vertical asymptote detection
        is_zero = False

    if xopt is not None:
        evals = floquet_basis.quasienergies(xopt).copy()
        omegad = 2*np.pi/floquet_basis.T

        evals_gap = evals[[[0], [1]]] - evals[None]
        evals_gap[[0, 1], [0, 1]] = np.nan
        # XXX: Assume time-reversal symmetry, so we wrap quasienergies around by 2*omegad.
        evals_gap = np.abs((evals_gap + omegad) % (2*omegad) - omegad)

        if np.any(evals_gap <= minimum_energy_gap):
            is_zero = False

    return xopt, is_zero


def calculate_polarization_and_error(
        qubit: Basis,
        frame: FloquetEigResult,
):
    EC = qubit.get_parameter('EC').item()
    n_op = qubit.get_operator('charge')
    nF_op = frame.dress_fft(n_op)

    kw, ii, jj = np.meshgrid(
        frame.fft_freqs,
        range(frame.freqs.shape[-1]),
        range(frame.freqs.shape[-1]),
        indexing='ij',
    )

    # Initial energy minus final energy.
    delta = frame.freqs[..., jj] - frame.freqs[..., ii]

    # Insert extra axis at the start to normalize non-perturbative arguments.
    if len(nF_op.shape) == 3:
        nF_op = nF_op[None]

    if len(delta.shape) == 3:
        delta = delta[None]

    delta[0] -= kw

    abs2_nF = polymul(nF_op.conj(), nF_op).real
    nF_diff = nF_op[..., 1, 1] - nF_op[..., 0, 0]
    abs2_nF_diff = polymul(nF_diff.conj(), nF_diff).real

    A = 16*EC * DIELECTRIC_LOSS_TANGENT
    transition_rates = np.sum(
        abs2_nF * (A * (delta[0] > 0))[None],
        axis=-3,
    )
    dephasing_rate = np.sum(
        0.5*abs2_nF_diff * (A * (frame.fft_freqs > 0))[None],
        axis=-1,
    )

    gamma_1 = transition_rates[..., 0, 1] + transition_rates[..., 1, 0]
    gamma_phi = dephasing_rate
    gamma_leak = (transition_rates[..., 2:, 0] + transition_rates[..., 2:, 1]).sum(axis=-1)

    err = 2/5 * (gamma_1 + gamma_phi) + 1/2 * gamma_leak

    p0 = (nF_op[..., 1, 0, 0]/n_op[1, 0]).real
    p1 = (nF_op[..., 1, 1, 1]/n_op[1, 0]).real

    if len(frame.modes.shape) == 3:
        # Non-perturbative argument.
        return p0.item(), p1.item(), err.item()
    else:
        return p0, p1, err


def calculate_avoided_crossing_gap(
        floquet_basis: AdiabaticFloquetBasis,
        i: int,
        j: int,
        k: int,
        step_size: float,
        xtol: float,
        x0: float|None = None,
        x1: float|None = None,
):
    if x0 is None:
        x0 = max(step_size, floquet_basis.domain[0])
    if x1 is None:
        x1 = floquet_basis.domain[1]

    omegad = 2*np.pi/floquet_basis.T

    def func(amp):
        result = floquet_basis.query_perturbative(amp, order=1)
        gap = result.freqs[:, i] - result.freqs[:, j]
        gap[0] -= k*omegad
        if gap[0] < 0:
            gap *= -1
        return gap[1]

    x, _ = find_root(func, x0, x1, step_size, xtol)

    if x is not None:
        freqs = floquet_basis.quasienergies(x)
        gap = abs(freqs[i] - (freqs[j] + k*omegad))
    else:
        gap = None

    return x, gap


def calculate_quasienergies_and_avgenergies_around_ac(
        floquet_basis: AdiabaticFloquetBasis,
        amps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    drive_freq = 2*np.pi/floquet_basis.T
    quasienergies = np.array([
        floquet_basis.quasienergies(amp)
        for amp in amps
    ])
    avgenergies = np.array([
        floquet_basis.average_energies(amp)
        for amp in amps
    ])
    quasienergies %= drive_freq
    quasienergies = np.unwrap(quasienergies, axis=0, period=drive_freq)

    permutation = np.argsort(quasienergies, axis=1)
    permutation = np.take_along_axis(permutation, np.argsort(permutation[0])[None], axis=1)

    quasienergies = np.take_along_axis(quasienergies, permutation, axis=1)
    avgenergies = np.take_along_axis(avgenergies, permutation, axis=1)

    return quasienergies, avgenergies


def calculate_optimal_amplitude(
        fx: Fluxonium,
        floquet_basis: AdiabaticFloquetBasis,
        step_size: float,
        xtol: float = 1e-6,
        x1: float|None = None,
):
    amp_opt = float('nan')
    p0_opt= float('nan')
    p1_opt = float('nan')
    y_opt = float('nan')
    objective = float('nan')
    is_zero_opt = False

    amp_tmp = 0.0

    # Loop for a maximum of 100 iterations.
    for _ in range(100):
        amp_tmp, is_zero_tmp = calculate_critical_amplitude(
            fx,
            floquet_basis,
            step_size=step_size,
            xtol=xtol,
            minimum_energy_gap=0.0,
            x0=amp_tmp + xtol,
            x1=x1,
        )

        if amp_tmp is None: break

        p0_tmp, p1_tmp, y_tmp = calculate_polarization_and_error(
            fx,
            floquet_basis.query(amp_tmp),
        )
        objective_tmp = abs(p1_tmp - p0_tmp)/y_tmp

        # When driving below the qubit frequency, p0 > p1,
        # and when driving above, p1 > p0.
        # If this is not satisfied, then we have probably overshot the optimum point.
        #  if is_zero_tmp and ((drive_frequency > qubit_frequency) ^ (p1_tmp > p0_tmp)):
            #  break

        # Note: objective is initialized as nan, hence the not of an inequality.
        if not (objective > objective_tmp):
            amp_opt = amp_tmp
            is_zero_opt = is_zero_tmp
            p0_opt = p0_tmp
            p1_opt = p1_tmp
            y_opt = y_tmp
            objective = objective_tmp

    if not is_zero_opt:
        p0_opt = float('nan')
        p1_opt = float('nan')
        y_opt = float('nan')
        objective = float('nan')

    return (
        amp_opt,
        p0_opt,
        p1_opt,
        y_opt,
    )


def calculate_amplitude_for_deltap(
        qubit: Basis,
        floquet_basis: AdiabaticFloquetBasis,
        deltap_data: ArrayLike,
) -> NDArray[np.floating]:
    deltap_data = np.asarray(deltap_data)

    def fun(x: float) -> float:
        p0, p1, _ = calculate_polarization_and_error(
            qubit,
            floquet_basis.query(x),
        )
        return abs(p0 - p1)

    def dfun(x: float) -> float:
        p0, p1, _ = calculate_polarization_and_error(
            qubit,
            floquet_basis.query_perturbative(x, order=1),
        )
        y = p1[0] - p0[0]

        if abs(y) < 1e-4:
            # If deltap is really small, assume the gradient is positive.
            dabsy = abs(p1[1] - p0[1])
        else:
            dabsy = np.sign(y)*(p1[1] - p0[1])

        return dabsy

    intervals, ranges = get_monotonic_increasing_intervals(
        fun,
        dfun,
        x0=floquet_basis.domain[0],
        x1=floquet_basis.domain[1],
        step_size=0.05,
        xtol=1e-9,
    )

    amp_data: list[float] = []
    for deltap in deltap_data:
        if deltap == 0.0:
            amp_data.append(0)
            continue

        for i, r in enumerate(ranges):
            if deltap >= r[0] and deltap <= r[1]:
                # Found the interval containing the target deltap.
                break
        else:
            amp_data.append(np.nan)
            continue

        x: float = scipy.optimize.bisect(  # type: ignore
            lambda x: fun(x) - deltap,
            a=intervals[i][0],
            b=intervals[i][1],
        )
        amp_data.append(x)

    return np.array(amp_data, dtype=float)



def polymul(a, b):
    c = np.zeros(a.shape, dtype=np.promote_types(a.dtype, b.dtype))
    for n in range(a.shape[0]):
        c[n] = (a[n::-1] * b[:n+1]).sum(axis=0)
    return c


def reciprocal(p):
    """Return 1/p as a geometric series."""
    d = p.degree()
    normalized_p = p/p.coef[0]
    geom = np.ones(d+1)
    pinv = 1/p.coef[0] * np.polynomial.Polynomial(geom)(1-normalized_p).cutdeg(d)
    return pinv
