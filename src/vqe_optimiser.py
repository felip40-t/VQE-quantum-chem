""" Run the VQE algorithm to find the ground state energy of a molecule. """

import numpy as np
from scipy.optimize import minimize

from Ansatz import HEA
from Hamiltonian import build_reduced_hamiltonian

from qiskit_aer.primitives import EstimatorV2 as Estimator
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

exact_estimator = StatevectorEstimator()
energy_history = []
param_history = []

def vqe_objective(params: np.ndarray, ansatz: HEA, hamiltonian: SparsePauliOp, estimator) -> float:
    """Objective function for VQE optimization."""
    bound_circuit = ansatz.build().assign_parameters(params)
    pub = (bound_circuit, hamiltonian)
    job = estimator.run([pub])
    result = job.result()
    energy = result[0].data.evs.real

    energy_history.append(energy)
    param_history.append(params.copy())
    return float(energy)