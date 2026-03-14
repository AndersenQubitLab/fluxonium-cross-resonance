from injector import Injector

from fluxoniumcr.qubits.basis import Basis
from fluxoniumcr.qubits.product_basis import DressedProductBasis, ProductBasis
from fluxoniumcr.simulation.module import SimulationModule


def create_two_coupled_qubits(
        q0: Basis,
        q1: Basis,
        JC: float,
        truncated_dims: tuple[int, int],
        dt: float = 0.01,
) -> Injector:
    bare_basis = ProductBasis([q0, q1])

    H_q0 = bare_basis.get_operator({0: 'hamiltonian'})
    H_q1 = bare_basis.get_operator({1: 'hamiltonian'})
    n0n1 = bare_basis.get_operator(['charge', 'charge'])

    H = H_q0 + H_q1 + JC*n0n1

    basis = DressedProductBasis(
        hamiltonian=H,
        bare_basis=bare_basis,
        truncated_dims=truncated_dims,
    )

    root = Injector([
        SimulationModule(
            basis=basis,
            names=['q0', 'q1'],
            dt=dt,
        ),
    ])

    return root
