from __future__ import annotations

from math import pi

import numpy as np
from numpy.typing import NDArray
import scipy.optimize


def calculate_cnot_fidelity(M: NDArray, apply_corrections: bool = True) -> NDArray:
    if apply_corrections:
        theta, phi = calculate_cnot_offset(M)
    else:
        theta = 0.0
        phi = 0.0

    U = create_cross_resonance_unitary(pi/2, theta - pi/2, phi - pi/2)

    # Equation (42) in:
    # Tripathi, V., Khezri, M. & Korotkov, A. N.
    # Operation and intrinsic error budget of a two-qubit cross-resonance gate.
    # Phys. Rev. A 100, 012301 (2019).
    fidelity = 1/20 * np.real(np.sum(M.conj() * M) + np.abs(np.sum(M.conj() * U))**2)

    return fidelity


def calculate_cnot_offset(M: np.ndarray) -> tuple[float, float]:
    """Calculate and return the IX rotation correction (theta) and the ZI rotation correction (phi)."""

    # Matrix elements of M in the (Z, X) basis.
    m0p = 0.5*(M[0,0] + M[0,1] + M[1,0] + M[1,1])
    m0m = 0.5*(M[0,0] - M[0,1] - M[1,0] + M[1,1])
    m1p = 0.5*(M[2,2] + M[2,3] + M[3,2] + M[3,3])
    m1m = 0.5*(M[2,2] - M[2,3] - M[3,2] + M[3,3])

    def fobj(phi):
        m1p_new = m1p*np.exp(-1j*phi)
        m1m_new = m1m*np.exp(-1j*phi)

        zp_magn = abs(m0p + m1p_new)
        zm_magn = abs(m0m - m1m_new)

        zp_magn_1 = -np.imag(m0p*m1p_new.conj())/zp_magn
        zm_magn_1 = np.imag(m0m*m1m_new.conj())/zm_magn

        zp_magn_2 = -1/zp_magn * (zp_magn_1**2 + np.real(m0p*m1p_new.conj()))
        zm_magn_2 = -1/zm_magn * (zm_magn_1**2 - np.real(m0m*m1m_new.conj()))

        zp_magn_3 = -zp_magn_1 * (1 + 3*zp_magn_2/zp_magn)
        zm_magn_3 = -zm_magn_1 * (1 + 3*zm_magn_2/zm_magn)

        f = zp_magn_1 + zm_magn_1
        df = zp_magn_2 + zm_magn_2
        ddf = zp_magn_3 + zm_magn_3

        return f, df, ddf

    phi_guess = np.angle(-m1m*m0m.conj()) + 0.5*np.angle(-m1p*m1m.conj()*m0p.conj()*m0m)

    result = scipy.optimize.root_scalar(
        fobj,
        x0=np.float64(phi_guess),  # Use double precision.
        fprime2=True,
        method='halley',
    )
    phi_opt = np.fmod(result.root, 2*pi)

    zp_opt = m0p + m1p * np.exp(-1j*phi_opt)
    zm_opt = m0m - m1m * np.exp(-1j*phi_opt)
    theta_opt = np.angle(zm_opt*zp_opt.conj())

    return theta_opt, phi_opt


PauliI = np.array([
    [1, 0],
    [0, 1],
])
PauliX = np.array([
    [0, 1],
    [1, 0],
])
PauliY = np.array([
    [  0, -1j],
    [ 1j,   0],
])
PauliZ = np.array([
    [ 1,  0],
    [ 0, -1],
])


def create_cross_resonance_unitary(
        chi: float,
        theta: float,
        phi: float,
) -> NDArray:
    cII = 0.5 * (
        np.exp(0.5j * phi) * np.cos((theta - chi)/2)
        + np.exp(-0.5j * phi) * np.cos((theta + chi)/2)
    )
    cZI = -0.5 * (
        np.exp(0.5j * phi) * np.cos((theta - chi)/2)
        - np.exp(-0.5j * phi) * np.cos((theta + chi)/2)
    )
    cIX = -0.5j * (
        np.exp(0.5j * phi) * np.sin((theta - chi)/2)
        + np.exp(-0.5j * phi) * np.sin((theta + chi)/2)
    )
    cZX = 0.5j * (
        np.exp(0.5j * phi) * np.sin((theta - chi)/2)
        - np.exp(-0.5j * phi) * np.sin((theta + chi)/2)
    )
    Ucr = (
        cII * np.kron(PauliI, PauliI)
        + cZI * np.kron(PauliZ, PauliI)
        + cIX * np.kron(PauliI, PauliX)
        + cZX * np.kron(PauliZ, PauliX)
    )
    return Ucr
