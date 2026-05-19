""" Script to reconstruct the density matrix from measurement results of pauli operators. """

import numpy as np
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit_ibm_runtime import EstimatorV2, Options
from DensityMatrix import DensityMatrix
from hamiltonian import map_fermionic_op_to_qubit_op, qubit_op_to_matrix

LABELS = ['I', 'X', 'Y', 'Z']

def construct_labels(num_qubits: int) -> list[str]:
    """Construct the labels for the Pauli operators."""
    if num_qubits == 1:
        return LABELS
    else:
        return [a + b for a in construct_labels(num_qubits - 1) for b in LABELS]

def measure_exp_vals(estimator: EstimatorV2, circuit: QuantumCircuit, num_qubits: int) -> np.ndarray:
    """Measure the expectation values of all Pauli operators."""
    labels = construct_labels(num_qubits)
    pauli_ops = [SparsePauliOp(label) for label in labels]
    job = estimator.run([(circuit, pauli_ops)])
    result = job.result()[0].data.evs
    std = job.result()[0].data.stds
    return np.array(result), np.array(std)

def reconstruct_density_matrix(exp_vals: np.ndarray, num_qubits: int) -> DensityMatrix:
    """Reconstruct the density matrix from the expectation values."""
    dim = 2 ** num_qubits
    labels = construct_labels(num_qubits)
    pauli_ops = [SparsePauliOp(label) for label in labels]
    rho = np.zeros((dim, dim), dtype=complex)
    for exp_val, pauli in zip(exp_vals, pauli_ops):
        rho += exp_val * pauli.to_matrix()
    return DensityMatrix(rho / dim)

def build_1rdm_operators(num_qubits: int, n_electrons: int) -> list[SparsePauliOp]:
    """Build the list of 1-RDM operators in the qubit basis."""
    operators = {}
    for p in range(num_qubits):
        for q in range(num_qubits):
            op = map_fermionic_op_to_qubit_op(p, q, num_qubits, n_electrons)
            operators[(p, q)] = op
    return operators

def reconstruct_1rdm(rho: DensityMatrix, num_qubits: int, n_electrons: int) -> DensityMatrix:
    """Reconstruct the 1-RDM from the full density matrix."""
    rdm = np.zeros((num_qubits, num_qubits), dtype=complex)
    operators = build_1rdm_operators(num_qubits, n_electrons)
    for (p, q), op in operators.items():
        op_matrix = qubit_op_to_matrix(op, num_qubits)
        rdm[p, q] = np.trace(rho.matrix @ op_matrix)
    return DensityMatrix(rdm)
